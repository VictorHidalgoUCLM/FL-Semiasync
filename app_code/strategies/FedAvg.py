import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from logging import INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import toml
import flwr as fl
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app_code.strategies.data_analyst import DataAnalyst


class ExportThread(threading.Thread):
    """A thread class responsible for periodically executing queries and exporting data
    using the provided DataAnalyst instance.
    """

    def __init__(
        self, analyst_instance: DataAnalyst, sleep_time: int, init_time: float
    ):
        """Initialize the ExportThread.

        Args:
            analyst_instance: An instance of the DataAnalyst class to execute queries and export data.
            sleep_time: Time interval (in seconds) between consecutive executions of queries.
            init_time: The initial timestamp to set for the analyst instance.
        """

        super().__init__()
        self.analyst_instance = analyst_instance
        self.sleep_time = sleep_time
        self.init_time = init_time

    def run(self):
        """The main logic of the thread. Executes recursive queries and exports data
        at regular intervals defined by sleep_time.
        """

        try:
            # Set the initial timestamp for the analyst instance
            self.analyst_instance.init_time = self.init_time

            # Continuously execute queries and export data
            while True:
                self.analyst_instance.execute_recursive_queries()  # Execute recursive queries
                self.analyst_instance.export_data()  # Export the collected data
                time.sleep(
                    self.sleep_time
                )  # Wait for the specified interval before repeating

        except IndexError as error:
            # Handle IndexError exceptions and print an error message
            print("Error in run method:", error)


class FedAvgCustom(FedAvg):
    def __init__(
        self,
        num_exec: int,
        strategy_name: str,
        *args: Any,
        debug: bool = False,
        **kwargs: Optional[Dict[str, Any]],
    ):
        """Initialize the FedAvgCustom strategy.

        Args:
            num_exec: Execution number for logging and tracking purposes.
            strategy_name: Name of the strategy being used.
            debug: Boolean to enable or disable (T/F) debug mode.
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(*args, **kwargs)

        # Initialize instance variables
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.init_time = 0  # Record the initialization time
        self.client_mapping = {}  # Mapping of client IDs to user IDs
        self.debug = debug  # Debug mode flag

        # Round offset for custom round numbering
        self.round_offset = 0

        # Load configuration from the TOML file specified in the environment variable
        self.config = toml.load(os.environ.get("CONFIG_PATH"))

        # Extract Prometheus and federation configurations
        self.prometheus_conf = self.config.get("prometheus_conf", {})
        self.federation_conf = self.config[self.config["tempConfig"]["federation"]]        

        # Define epochs and metrics for tracking
        self.epochs = ["fit", "ev"]
        self.metrics = ["accuracy", "loss", "precision", "recall", "f1"]

        # Initialize data and timestamps dictionaries
        self.data = {}
        self.timestamps = {}

        # Populate data and timestamps structures for server for each epoch
        self.timestamps.setdefault("server", {})
        self.data.setdefault("server", {})

        self.nodes = ["server"]

        for epoch in self.epochs:
            self.timestamps["server"].setdefault(epoch, {})
            self.data["server"].setdefault(
                epoch, {metric: 0 for metric in self.metrics}
            )

    def set_round_offset(self, offset: int):
        """Set the round offset for the current instance.

        Args:
            offset: The offset value to set for the round.
        """

        self.round_offset = offset

    def process_client(
        self,
        client: ClientProxy,
        fit_ins: FitIns,
        parameters_fit: Optional[dict],
        server_round: int,
    ):
        """Process a client by setting its configuration and mapping its ID.

        Args:
            client: The client to process.
            fit_ins: The fit instructions to configure.
            parameters_fit: Additional parameters for the fit configuration.
            server_round: The current server round.
        """    
        if server_round == 1:
            # Retrieve client properties during the first round
            prop = client.get_properties(
                GetPropertiesIns({}), timeout=None, group_id=None
            )
            if prop.properties["user"]:
                # Map the client ID to the user ID
                id = prop.properties["user"]
                self.client_mapping[client.cid] = id

            # Get the client-specific configuration from the loaded configuration
            client_conf = self.config["names"].get(id)

            # Set the fit instructions configuration
            fit_ins.config = {
                "epochs": client_conf[0],
                "batch_size": client_conf[1],
                "subset_size": client_conf[2],
                "server_round": server_round,
                "debug": self.debug,
            }

            # Add any additional parameters to the fit configuration
            if parameters_fit is not None:
                for key, value in parameters_fit.items():
                    fit_ins.config[key] = value

            # Retrieve the mapped user ID for the client
            id = self.client_mapping[client.cid]

            self.nodes.append(id)

            self.timestamps.setdefault(id, {})
            self.data.setdefault(id, {})

            for epoch in self.epochs:
                self.timestamps[id].setdefault(epoch, {})
                self.data[id].setdefault(
                    epoch, {metric: 0 for metric in self.metrics}
                )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_fit: Optional[dict] = None,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Get client/config standard pairs from the FedAvg superclass
        fit_config = super().configure_fit(server_round, parameters, client_manager)

        # Use a ThreadPoolExecutor to process clients concurrently
        # This processing is made because it is slow secuentially
        with ThreadPoolExecutor() as executor:
            for client, fit_ins in fit_config:
                executor.submit(
                    self.process_client, client, fit_ins, parameters_fit, server_round
                )

        # Initialize the data analyst and start the export thread during the first round
        if server_round == 1 and not self.debug:
            self.init_time = time.time()  # Initial time once first config has been done

            analyst = DataAnalyst(
                self.prometheus_conf["prometheus_url"],
                self.num_exec,
                self.strategy_name,
            )  # Create an instance of DataAnalyst with Prometheus configuration

            # Retrieve hostnames and set up queries
            analyst.get_hostnames()
            analyst.create_queries()
            analyst.clients_up()

            # Execute one-time queries for initial data collection
            analyst.execute_one_time_queries()

            # Start a background thread for periodic data export
            export_thread = ExportThread(
                analyst, self.prometheus_conf["sleep_time"], self.init_time
            )
            export_thread.daemon = (
                True  # Ensure the thread exits when the main program ends
            )
            export_thread.start()

        return fit_config, self.client_mapping

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        counters: List[int],
        times: Dict[str, List[float]] = {}
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results from clients and save aggregated parameters periodically.

        Args:
            server_round: The current server round.
            results: A list of tuples containing client proxies and their fit results.
            failures: A list of failures during the fit process.
            times: A dictionary mapping client IDs to their respective timing information.

        Returns:
            A tuple containing the aggregated parameters and metrics.
        """
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Extract the config parameters
        temp_config = self.config["tempConfig"]
        checkpoint_path = self.config["paths"]["checkpoint"]

        # Format the checkpoint path with the extracted values
        formatted_path = checkpoint_path.format(
            strategy=self.strategy_name,
            num_exec=temp_config["num_exec"],
            federation=temp_config["federation"],
            sub_execution=temp_config["execution_name"],
        )

        # Expand user directory and store the result
        directory_name = os.path.expanduser(formatted_path)

        # Create the directory if it does not exist
        os.makedirs(directory_name, exist_ok=True)

        # Write client-specific metrics to self.data from fit
        for client, fit_res in results:
            id = self.client_mapping[client.cid]
            self.timestamps[id]["fit"] = ";".join(map(str, times[id]))

            for metric in self.metrics:
                self.data[id]["fit"][metric] = fit_res.metrics[metric]

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated parameters every 5 rounds or at the last round
            if (
                server_round % 5 == 0
                or server_round + self.round_offset == self.config["config"]["rounds"]
            ):
                log(INFO, f"")
                log(INFO, f"Saving round {server_round} aggregated_ndarrays...")
                log(INFO, f"")

                # Save the aggregated arrays to the file
                filename = f"{directory_name}/round-{server_round + self.round_offset}-weights.npz"
                np.savez(filename, *aggregated_ndarrays)

            # Write server-specific metrics to self.data
            for metric in self.metrics:
                self.data["server"]["fit"][metric] = aggregated_metrics[metric]

            # Record the timestamp for the server's fit process
            self.timestamps["server"]["fit"] = time.time() - self.init_time

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        times: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients and save aggregated metrics periodically.

        Args:
            server_round: The current server round.
            results: A list of tuples containing client proxies and their fit results.
            failures: A list of failures during the fit process.
            times: A dictionary mapping client IDs to their respective timing information.

        Returns:
            A tuple containing the aggregated loss and metrics.
        """

        # Call aggregate_evaluate from base class (FedAvg) to aggregate parameters and metrics
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Extract the necessary parameters for the log path
        temp_config = self.config["tempConfig"]
        log_path = self.config["paths"]["logs"]

        # Format the log path with the extracted values
        formatted_log_path = log_path.format(
            strategy=self.strategy_name,
            federation=temp_config["federation"],
            sub_execution=temp_config["execution_name"],
        )

        # Expand user directory and store the result
        directory_name = os.path.expanduser(formatted_log_path)

        # Create the directory if it does not exist
        os.makedirs(directory_name, exist_ok=True)

        # Write client-specific metrics to self.data from evaluate
        for client, evaluate_res in results:
            id = self.client_mapping[client.cid]
            self.timestamps[id]["ev"] = times[id]

            for metric in self.metrics:
                if metric == "loss":
                    self.data[id]["ev"][metric] = evaluate_res.loss
                else:
                    self.data[id]["ev"][metric] = evaluate_res.metrics[metric]

        # If metrics are not empty
        if loss_aggregated is not None and metrics_aggregated is not None:
            for metric in self.metrics:
                if metric == "loss":
                    self.data["server"]["ev"][metric] = loss_aggregated
                else:
                    self.data["server"]["ev"][metric] = metrics_aggregated[metric]

            # Record the timestamp for the server's ev process
            self.timestamps["server"]["ev"] = time.time() - self.init_time

            flattened_logs = {}  # Dictionary for flattening metrics
            flattened_timestamps = {}  # Dictionary for flatenning timestamps

            # Flattening all data collected
            for node in self.nodes:
                for epoch in self.epochs:
                    flattened_timestamps[f"{node}_{epoch}"] = self.timestamps[node][epoch]
                    for metric in self.metrics:
                        flattened_logs[f"{node}_{epoch}_{metric}"] = self.data[node][epoch][metric]

            # Open log file and write metrics
            with open(
                f"{directory_name}/log_{self.num_exec}.csv", mode="a", newline=""
            ) as file_logs, open(
                f"{directory_name}/timestamp_{self.num_exec}.csv", mode="a", newline=""
            ) as file_timestamps:

                writer_csv_logs = csv.writer(file_logs)
                writer_csv_timestamps = csv.writer(file_timestamps)

                # Write column names for metrics if file is empty
                if file_logs.tell() == 0:
                    writer_csv_logs.writerow(list(flattened_logs.keys()))
                if file_timestamps.tell() == 0:
                    writer_csv_timestamps.writerow(list(flattened_timestamps.keys()))

                # Write metrics values
                writer_csv_logs.writerow(list(flattened_logs.values()))
                writer_csv_timestamps.writerow(list(flattened_timestamps.values()))

        return loss_aggregated, metrics_aggregated
