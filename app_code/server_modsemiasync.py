import concurrent.futures
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union

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

class CustomServerModSemiasync(Server):
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
        self.inner_rounds = (
            {}
        )  # Dictionary that stores inner round for each client's id

        self.client_dictionary = {}
        self.projectconf = toml.load(os.environ.get("CONFIG_PATH"))

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
        ] = self.strategy.aggregate_fit(
                server_round=server_round,
                results=results,
                failures=failures,
                counters=counters,
                w_global=w_global,
                times=times
            )

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
        m = self.strategy.m     # Get semiasynchrony level from strategy
 
        # Initialize client_data structure
        for client, ins in client_instructions:
            client_id = self.client_mapping[client.cid]

            client_data = self.client_dictionary.setdefault(
                client_id, {"future": None, "counter": 0, "inner_round": 1, "times": []}
            )

            client_data["future"] = None
            client_data["counter"] = 0
            client_data["times"] = []

        while any(client["counter"] == 0 for client in self.client_dictionary.values()):
            completed = 0
            finished_fs = []

            for client, ins in client_instructions:
                client_id = self.client_mapping[client.cid]
                client_data = self.client_dictionary[client_id]

                if client_data["future"] is None:
                    if parameters_aggregated is not None:
                        ins.parameters = parameters_aggregated

                    ins.config["inner_round"] = client_data["inner_round"]

                    future = self.executor.submit(
                        fit_client, client, ins, timeout, group_id
                    )
                    client_data["future"] = future

            valid_futures = _get_valid_futures(self.client_dictionary)

            for future in as_completed(valid_futures.values()):
                client_id = next(id for id, f in valid_futures.items() if f == future)
                client_data = self.client_dictionary[client_id]

                with self.lock:
                    completed += 1
                    _update_client_data(client_data=client_data, init_time=self.init_time)

                    finished_fs.append(
                        (future, self.client_dictionary[client_id]["counter"], client_id, True)
                    )

                if completed == m:
                    break

            inter_results: list[tuple[ClientProxy, FitRes, str]] = []
            inter_failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
            inter_counters: list[int] = []
            for future, counter, client_id, inner_round in finished_fs:
                _handle_finished_future_after_fit(
                    future=future,
                    results=inter_results,
                    failures=inter_failures,
                    counters=inter_counters,
                    counter=counter,
                    client_id=client_id,
                    inner_round=inner_round
                )

            # Check if there are any inter_results (intermediate aggregation)
            filtered_results = [
                (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
                for _, fitres, _, _ in inter_results
            ]

            aggregated_ndarrays = aggregate(filtered_results)  
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
            
            # Calculate total data processed
            total_data = 0
            for inter_result in inter_results:
                total_data += inter_result[1].num_examples

            # For each inter_result store historic data for FedMOpt
            loss_list = []
            weight_list = []

            for inter_result in inter_results:
                key = inter_result[2]
                loss_value = inter_result[1].metrics['loss']
                loss_list.append(loss_value)
                weight_list.append(inter_result[1].num_examples / total_data)

                if key not in self.strategy.prev_loss:
                    self.strategy.prev_loss[key] = loss_value

                mejora_relativa = (self.strategy.prev_loss[key] - loss_value) / self.strategy.prev_loss[key]

                if hasattr(self.strategy, 'data_history'):
                    if key in self.strategy.data_history:
                        # Append al arreglo existente
                        self.strategy.data_history[key].append(mejora_relativa)
                    else:
                        # Crear nueva lista con el primer valor
                        self.strategy.data_history[key] = [mejora_relativa]

                self.strategy.prev_loss[key] = loss_value

        valid_futures = _get_valid_futures(self.client_dictionary)

        # Wait for executing clients
        for future in as_completed(valid_futures.values()):
            client_id = next(id for id, f in valid_futures.items() if f == future)
            client_data = self.client_dictionary[client_id]

            with self.lock:
                _update_client_data(client_data=client_data, init_time=self.init_time)

                finished_fs.append(
                        (future, client_data["counter"], client_id, False)
                )

        # Gather results
        results: list[tuple[ClientProxy, FitRes, str]] = []
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
        counters: list[int] = []
        for future, counter, client_id, inner_round in finished_fs:
            _handle_finished_future_after_fit(
                future=future,
                results=results,
                failures=failures,
                counters=counters,
                counter=counter,
                client_id=client_id,
                inner_round=inner_round
            )

        for result in results:
            inner_round = result[3]

            if not inner_round:                
                key = result[2]
                loss_value = result[1].metrics['loss']
                loss_list.append(loss_value)
                weight_list.append(inter_result[1].num_examples / total_data)

                if key not in self.strategy.prev_loss:
                    self.strategy.prev_loss[key] = loss_value

                mejora_relativa = (self.strategy.prev_loss[key] - loss_value) / self.strategy.prev_loss[key]

                if hasattr(self.strategy, 'data_history'):
                    if key in self.strategy.data_history:
                        # Append al arreglo existente
                        self.strategy.data_history[key].append(mejora_relativa)
                    else:
                        # Crear nueva lista con el primer valor
                        self.strategy.data_history[key] = [mejora_relativa]

                self.strategy.prev_loss[key] = loss_value

        results = [(client, fitres) for client, fitres, _, _ in results]

        """
        # Early stopping logic
        exceptions = 0
        for failure in failures:
            if isinstance(failure, ValueError):
                exceptions += 1

        if exceptions == len(finished_fs):
            print("Disconnecting")
            self.disconnect_all_clients(timeout=15.0)"""

        return (
            results,
            failures,
            {
                client_id: client_data["times"]
                for client_id, client_data in self.client_dictionary.items()
            },
            counters,
        )


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes, str]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    counters: List[int],
    counter: int,
    client_id: str,
    inner_round: bool,
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
    results.append((temp, res, client_id, inner_round))
    counters.append(counter)

def _update_client_data(
    client_data: Dict[str, Any],
    init_time: float
):
    """Updates necessary client data."""
    client_data["future"] = None
    client_data["counter"] += 1
    client_data["inner_round"] += 1
    client_data["times"].append(time.time() - init_time)


def _get_valid_futures(
    client_dictionary: Dict[str, Any]
):
    """Returns updated valid futures that are still running."""
    return {
        client_id: client_data["future"]
        for client_id, client_data in client_dictionary.items()
        if client_data["future"] is not None
    }

