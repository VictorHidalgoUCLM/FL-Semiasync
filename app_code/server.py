import concurrent.futures
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import INFO
from typing import Any, Dict, List, Optional, Tuple, Union

import toml
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
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
from flwr.server.strategy.aggregate import aggregate_inplace


class CustomServer(Server):
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
        self.time = time.time()  # Saves init time of class to use as reference
        self.inner_rounds = (
            {}
        )  # Dictionary that stores inner round for each client's id

        self.client_dictionary = {}

        self.projectconf = toml.load(os.environ.get("CONFIG_PATH"))

    def map_clients(self, group_id, client_instructions):
        """Map client cid with client's id (name)"""
        if group_id == 1:
            with ThreadPoolExecutor() as executor_prop:
                for client, _ in client_instructions:
                    future_prop = executor_prop.submit(
                        client.get_properties,
                        GetPropertiesIns({}),
                        timeout=None,
                        group_id=10,
                    )
                    prop = future_prop.result()
                    if prop.properties["user"]:
                        client_id = prop.properties["user"]
                        self.client_mapping[client.cid] = client_id  # Association

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
                times[client_id] = time.time() - self.time
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
        parameters_aggregated = None  # Parameters calculated between subrounds

        m = self.projectconf["synchrony"]  # Get semi-asynchrony level (M)

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
                    _update_client_data(client_data=client_data, init_time=self.time)

                    finished_fs.append(
                        (future, self.client_dictionary[client_id]["counter"])
                    )

                if completed == m:
                    break

            inter_results: list[tuple[ClientProxy, FitRes]] = []
            inter_failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
            for future, counter in finished_fs:
                _handle_finished_future_after_fit(
                    future=future,
                    results=inter_results,
                    failures=inter_failures,
                    counter=counter,
                    alpha=0.5,
                )

            aggregated_ndarrays = aggregate_inplace(inter_results)
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        valid_futures = _get_valid_futures(self.client_dictionary)

        # Wait for executing clients
        for future in as_completed(valid_futures.values()):
            client_id = next(id for id, f in valid_futures.items() if f == future)
            client_data = self.client_dictionary[client_id]

            with self.lock:
                _update_client_data(client_data=client_data, init_time=self.time)

                finished_fs.append(
                        (future, self.client_dictionary[client_id]["counter"])
                )

        # Gather results
        results: list[tuple[ClientProxy, FitRes]] = []
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
        for future, counter in finished_fs:
            _handle_finished_future_after_fit(
                future=future,
                results=results,
                failures=failures,
                counter=counter,
                alpha=0.5,
            )
        return (
            results,
            failures,
            {
                client_id: client_data["times"]
                for client_id, client_data in self.client_dictionary.items()
            },
        )


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    counter: int,
    alpha: float,
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
    _, res = result

    penalty = 1 / (1 + alpha * (counter - 1))
    normal_penalty = max(penalty, 0.4)

    res.num_examples *= normal_penalty

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


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