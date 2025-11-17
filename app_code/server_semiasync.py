import concurrent.futures
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import toml
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import (
    Server,
    _handle_finished_future_after_evaluate,
    evaluate_client,
    fit_client,
    EvaluateResultsAndFailures,
    FitResultsAndFailures,
)
from flwr.server.strategy.aggregate import aggregate
import timeit
from app_code.history import MyHistory

class CustomServerSemiasync(Server):
    """Class that inherits from Flwr original Server class. This permits to change the synchrony
    behaviour that default FL on Flwr defines.
    """

    def __init__(self, strategy, *args, **kwargs):
        """Inits CustomServer class.

        Args:
            strategy: object that defines the strategy FL will use.
        """
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.executor = ThreadPoolExecutor(
            self.max_workers
        )  # Thread pool to monitorize each client with a thread. Used for semi-asyncrhonous behaviour in FL.
        self.lock = threading.Lock()  # Lock defined for managing each monitoring thread
        self.futures = {}  # Dictionary that associates client's id with threads
        self.client_mapping = (
            {}
        )  # Dictionary that associates client identifier with client names
        self.init_time = 0  # Saves init time of class to use as reference

        self.client_dictionary = {}
        self.projectconf = toml.load(os.environ.get("CONFIG_PATH"))

        self.init_loss = 0
        self.latest_round = 1
        self.parameters_history = {}
        self.angles_history = {}

        self.prev_grad = None

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
        )  # Gets 'times', which are timestamps of each client
        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        try:
            # Aggregate the evaluation results
            aggregated_result: tuple[
                Optional[float],
                dict[str, Scalar],
            ] = self.strategy.aggregate_evaluate(server_round, results, failures, times)

        except Exception:
            raise

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
        # Processing each client that performs evaluation
        for client, ins in client_instructions:
            client_id = self.client_mapping[client.cid]

            # Submit future monitoring task associated to a client
            future = self.executor.submit(
                evaluate_client, client, ins, timeout, group_id
            )
            self.futures[client_id] = future

        # Extract valid futures so that as_completed can be used
        valid_futures = {
            key: future for key, future in self.futures.items() if future is not None
        }

        for future in as_completed(valid_futures.values()):
            # As soon as a future ends (client ends evaluation), a new for loop starts
            client_id = next(
                id for id, f in self.futures.items() if f == future
            )  # Gets current client_id

            with self.lock:  # Uses lock to update shared variables
                times[client_id] = time.time() - self.init_time
                finished_fs[client_id] = future  # New finished future
                self.futures[client_id] = None  # Clears future asociated to a client

        # Gather results
        results: list[tuple[ClientProxy, EvaluateRes]] = []
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]] = []
        for _, future in finished_fs.items():
            _handle_finished_future_after_evaluate(
                future=future, results=results, failures=failures
            )

        return results, failures, times

    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[MyHistory, float]:
        """Run federated averaging for a number of rounds."""
        history = MyHistory()

        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            self.init_loss = res[0]
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        # Run federated learning for num_rounds
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
                w_global=self.parameters,
            )
            
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            try:
                res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)

            except Exception as e:
                log(WARNING, "")
                log(WARNING, f"{e}")
                log(WARNING, f"Charging latest saved model and reevaluating...")

                if self.strategy.best_parameters is not None:
                    self.parameters = self.strategy.best_parameters

                    res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)

                    loss_fed, evaluate_metrics_fed, _ = res_fed
                    history.add_loss_distributed(
                            server_round=current_round, loss=loss_fed
                        )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

                break
            
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

            if self.strategy.strategy_name == "FedMOpt":
                history.add_fedMOpt(server_round=current_round,
                                    optimized_M=self.strategy.m,
                                    utility_list=self.strategy.utilities,
                                    time_list=self.strategy.times
                                )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        return history, elapsed

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
        w_global: Optional[Parameters],
    ) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions, self.client_mapping = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
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

        if server_round == 1:
            self.init_time = self.strategy.init_time
            
        self.parameters_history[server_round] = self.parameters

        # Collect `fit` results from all clients participating in this round
        results, failures, times, counters = self.fit_clients(
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
            dict[str, float],
            Optional[np.ndarray],
        ] = self.strategy.aggregate_fit(
                server_round=server_round,
                results=results,
                failures=failures,
                counters=counters,
                w_global=w_global,
                times=times,
                parameters_history=self.parameters_history,
                angles_history=self.angles_history,
                prev_grad=self.prev_grad
            )

        results = [(client, fitres) for client, fitres, _, _ in results]
        parameters_aggregated, metrics_aggregated, angles_history, g_final = aggregated_result
        
        self.angles_history = angles_history
        self.prev_grad = g_final

        keys_to_delete = [k for k in self.parameters_history if k < self.latest_round]
        for k in keys_to_delete:
            del self.parameters_history[k]

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def fit_clients(
        self,
        client_instructions: list[tuple[ClientProxy, FitIns]],
        max_workers: Optional[int],
        timeout: Optional[float],
        group_id: int,
    ) -> FitResultsAndFailures:
        """Refine parameters concurrently on all selected clients."""
        m = self.strategy.m     # Get semiasynchrony level from strategy
        completed = 0
        finished_fs = []

        if group_id == 1:
            for client, _ in client_instructions:
                client_id = self.client_mapping[client.cid]

                client_data = self.client_dictionary.setdefault(
                    client_id, {"future": None, "counter": 0, "times": [], "init_round": 1}
                )

        for client, ins in client_instructions:
            client_id = self.client_mapping[client.cid]
            client_data = self.client_dictionary[client_id]

            if client_data["future"] is None:
                future = self.executor.submit(fit_client, client, ins, timeout, group_id)
                client_data["future"] = future
                client_data["init_round"] = group_id

        group_ids = [
            data["init_round"]
            for data in self.client_dictionary.values()
            if data.get("init_round") is not None
        ]

        # Calcular el mínimo
        self.latest_round = min(group_ids) if group_ids else None

        valid_futures = _get_valid_futures(self.client_dictionary)

        for future in as_completed(valid_futures.values()):
            client_id = next(id for id, f in valid_futures.items() if f == future)
            client_data = self.client_dictionary[client_id]

            with self.lock:
                completed += 1
                _update_client_data(client_data=client_data, init_time=self.init_time)
                finished_fs.append(
                    (future, client_data["counter"], client_id, client_data["init_round"])
                )

            if completed == m:
                break

        for cid, future in valid_futures.items():
            if cid not in [fut_cid for fut_cid, f in valid_futures.items() if f.done()]:
                self.client_dictionary[cid]["counter"] += 1

        return_counters = {cid: data["counter"] for cid, data in self.client_dictionary.items()}

        # Gather results
        results: list[tuple[ClientProxy, FitRes, str]] = []
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
        counters: list[int] = []
        for future, counter, client_id, init_round in finished_fs:
            _handle_finished_future_after_fit(
                future=future,
                results=results,
                failures=failures,
                counters=counters,
                counter=counter,
                client_id=client_id,
                init_round=init_round
            )

        """ Log writing """
        for result in results:
            key = result[2]
            loss_value = result[1].metrics['loss']

            if key not in self.strategy.prev_loss:
                self.strategy.prev_loss[key] = self.init_loss

            mejora_relativa = (self.strategy.prev_loss[key] - loss_value) / self.strategy.prev_loss[key]

            if hasattr(self.strategy, 'data_history'):
                if key in self.strategy.data_history:
                    self.strategy.data_history[key].append(mejora_relativa)
                else:
                    self.strategy.data_history[key] = [mejora_relativa]

            self.strategy.prev_loss[key] = loss_value

        #results = [(client, fitres) for client, fitres, _ in results]

        return (
            results,
            failures,
            {
                client_id: client_data["times"]
                for client_id, client_data in self.client_dictionary.items()
            },
            return_counters,
        )


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes, str]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    counters: List[int],
    counter: int,
    client_id: str,
    init_round: int,
) -> None:
    """Rewrites original: Convert finished future into either a result or a failure. Counter is used
    to take into account the number of times each client has executed before.
    """
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    temp, res = result

    # Check result status code
    if res.status.code != Code.OK:
        failures.append(result)
        return
    
    # Check if parameters are empty (no tensors)
    if not res.parameters.tensors:
        # You might optionally also update status here if desired
        failures.append(result)
        return

    # All good: store result and counter
    results.append((temp, res, client_id, init_round))
    counters.append(counter)

def _update_client_data(
    client_data: Dict[str, Any],
    init_time: float
):
    """Updates necessary client data."""
    client_data["future"] = None
    client_data["times"].append(time.time() - init_time)
    client_data["counter"] = 0

def _get_valid_futures(
    client_dictionary: Dict[str, Any]
):
    """Returns updated valid futures that are still running."""
    return {
        client_id: client_data["future"]
        for client_id, client_data in client_dictionary.items()
        if client_data["future"] is not None
    }
