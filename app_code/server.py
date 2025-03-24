from flwr.server.server import Server, fit_client, evaluate_client, _handle_finished_future_after_evaluate
from logging import INFO
from typing import Optional, Union, List, Tuple
import toml
import os

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    GetPropertiesIns,
    ndarrays_to_parameters
)
from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import threading
import time
from flwr.server.strategy.aggregate import aggregate_inplace

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[Union[tuple[ClientProxy, FitRes], BaseException]],
]

EvaluateResultsAndFailures = tuple[
    list[tuple[ClientProxy, EvaluateRes]],
    list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
]

class CustomServer(Server):
    def __init__(self, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.executor = ThreadPoolExecutor(self.max_workers)
        self.lock = threading.Lock()
        self.futures = {}
        self.client_mapping = {}
        self.time = time.time()
        self.inner_rounds = {}

        self.projectconf = toml.load(os.environ.get('CONFIG_PATH'))

    def map_clients(self, group_id, client_instructions):
        if group_id == 1:
            with ThreadPoolExecutor() as executor_prop:
                for client, _ in client_instructions:
                    future_prop = executor_prop.submit(client.get_properties, GetPropertiesIns({}), timeout=None, group_id=10)
                    prop = future_prop.result()
                    if prop.properties["user"]:
                        client_id = prop.properties["user"]
                        self.client_mapping[client.cid] = client_id

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[float], dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "configure_evaluate: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures, times = self.evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: tuple[
            Optional[float],
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures, times)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)


    def evaluate_clients(
        self,
        client_instructions: list[tuple[ClientProxy, EvaluateIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
        group_id: int,
    ) -> EvaluateResultsAndFailures:
        """Evaluate parameters concurrently on all selected clients."""
        times = {}
        finished_fs = {}

        for client, ins in client_instructions:
            client_id = self.client_mapping[client.cid]

            future = self.executor.submit(evaluate_client, client, ins, timeout, group_id)
            self.futures[client_id] = future

        # Extract valid futures so that as_completed can be used
        valid_futures = {key: future for key, future in self.futures.items() if future is not None}

        for future in as_completed(valid_futures.values()):
            client_id = next(id for id, f in self.futures.items() if f == future)

            # Lock the incrementation of the completed_count variable for every future
            with self.lock:
                times[client_id] = time.time() - self.time
                finished_fs[client_id] = future
                self.futures[client_id] = None

        # Gather results
        results: list[tuple[ClientProxy, EvaluateRes]] = []
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]] = []
        for _, future in finished_fs.items():
            _handle_finished_future_after_evaluate(
                future=future, results=results, failures=failures
            )

        return results, failures, times


    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        self.map_clients(server_round, client_instructions)

        # Collect `fit` results from all clients participating in this round
        results, failures, times = self.fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )

        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures, times)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def fit_clients(
        self,
        client_instructions: list[tuple[ClientProxy, FitIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
        group_id: int,
    ) -> FitResultsAndFailures:
        """Refine parameters concurrently on all selected clients."""
        counters = {}   # Diccionario para contabilizar ejecuciones por ronda de cada cliente
        parameters_aggregated = None    # Guardamos los parámetros agregados entre subrondas
        finished_fs = {}    # Diccionario que almacena los threads terminados
        times = {}      # Diccionario que almacena timestamps

        m = self.projectconf['synchrony']   # Esperamos a m clientes en la semiasincronía

        # Loop infinito dentro de una ronda, seguimos si hay al menos un cliente que no ha ejecutado 1 vez
        while True:
            completed_count = 0     # Cada subronda comienza con 0 elementos completados

            # Para cada cliente seleccionamos si lanzamos o no la tarea
            for client, ins in client_instructions:
                client_id = self.client_mapping[client.cid]     # Obtenemos el nombre del cliente

                # Si el cliente no tiene un contador asociado...
                if counters.get(client_id) is None:
                    counters[client_id] = 0     # Inicializamos el contador a 0
                    times[client_id] = []       # Inicializamos sus timestamps

                if self.inner_rounds.get(client_id) is None:
                    self.inner_rounds[client_id] = 1    # Inicializamos su ronda interna a 1

                # Si existe algún contador a 0...
                if any(value == 0 for value in counters.values()):
                    # y el cliente no tiene asociado un hilo de ejecución...
                    if self.futures.get(client_id) is None:
                        # Si se ha inicializado los parámetros agregados en una subronda anterior
                        if not parameters_aggregated is None:
                            ins.parameters = parameters_aggregated      # Se incluyen los parámetros agregados para entrenar localmente
                        
                        ins.config['inner_round'] = self.inner_rounds.get(client_id)       # Se actualiza el inner round del cliente

                        future = self.executor.submit(fit_client, client, ins, timeout, group_id)   # Se lanza el hilo
                        self.futures[client_id] = future    # Se asocia el hilo a un cliente

            # Determine wether server has to wait for M clients to finish or remaining clients
            n = m if any(value == 0 for value in counters.values()) else sum(1 for value in self.futures.values() if value is not None)

            # Extract valid futures so that as_completed can be used
            valid_futures = {key: future for key, future in self.futures.items() if future is not None}

            for future in as_completed(valid_futures.values()):
                client_id = next(id for id, f in self.futures.items() if f == future)

                # Lock the incrementation of the completed_count variable for every future
                with self.lock:
                    completed_count += 1
                    counters[client_id] += 1
                    self.inner_rounds[client_id] += 1

                    self.futures[client_id] = None
                    finished_fs[client_id] = (future, counters[client_id])

                    times[client_id].append(time.time() - self.time)

                if completed_count == n:
                    break

            if any(value == 0 for value in counters.values()):
                inter_results: list[tuple[ClientProxy, FitRes]] = []
                inter_failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
                for _, (future, counter) in finished_fs.items():
                    _handle_finished_future_after_fit(
                        future=future, results=inter_results, failures=inter_failures, counter=counter
                    )

                aggregated_ndarrays = aggregate_inplace(inter_results)
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

                finished_fs = {}

            if  n != m:
                break

        # Gather results
        results: list[tuple[ClientProxy, FitRes]] = []
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
        for _, (future, counter) in finished_fs.items():
            _handle_finished_future_after_fit(
                future=future, results=results, failures=failures, counter=counter
            )
        return results, failures, times
    

def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    counter: int
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    res.num_examples *= counter

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)