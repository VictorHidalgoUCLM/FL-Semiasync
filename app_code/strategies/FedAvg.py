import csv
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from logging import INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import math

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
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace

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
        **kwargs: Optional[Dict[str, Any]],
    ):
        """Initialize the FedAvgCustom strategy.

        Args:
            num_exec: Execution number for logging and tracking purposes.
            strategy_name: Name of the strategy being used.
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(*args, **kwargs)

        # Initialize instance variables
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.init_time = 0  # Record the initialization time
        self.client_mapping = {}  # Mapping of client IDs to user IDs

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
        self.prev_loss = {}
        self.prev_counters = {}

        # Populate data and timestamps structures for server for each epoch
        self.timestamps.setdefault("server", {})
        self.data.setdefault("server", {})

        self.filename = None

        self.nodes = ["server"]

        for epoch in self.epochs:
            self.timestamps["server"].setdefault(epoch, {})
            self.data["server"].setdefault(
                epoch, {metric: 0 for metric in self.metrics}
            )

        self.m = self.config["synchrony"]
        self.prop = {}

        self.server_type = self.config.get("config", {}).get("server_type", 1)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        loss, metrics = eval_res

        # Guardar en estructuras internas
        self.data["server"]["ev"]["loss"] = loss
        for metric in self.metrics:
            if metric != "loss" and metric in metrics:
                self.data["server"]["ev"][metric] = metrics[metric]

        # Timestamp de la evaluación en el servidor
        self.timestamps["server"]["ev"] = time.time() - self.init_time

        # Construir ruta de logs
        temp_config = self.config["tempConfig"]
        log_path = self.config["paths"]["logs"]
        formatted_log_path = log_path.format(
            strategy=self.strategy_name,
            federation=temp_config["federation"],
            sub_execution=temp_config["execution_name"],
        )
        directory_name = os.path.expanduser(formatted_log_path)
        os.makedirs(directory_name, exist_ok=True)

        # Escribir métricas al log completo
        if server_round > 0:
            self._write_logs(directory_name)

        return loss, metrics

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
                self.prop[id] = prop

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

        # Retrieve the mapped user ID for the client in case it was not done before
        id = self.client_mapping[client.cid]

        # Get the client-specific configuration from the loaded configuration
        client_conf = self.config["names"].get(id)

        # Set the fit instructions configuration
        fit_ins.config = {
            "epochs": client_conf[0],
            "batch_size": client_conf[1],
            "subset_size": client_conf[2],
            "server_round": server_round
        }

        # Add any additional parameters to the fit configuration
        if parameters_fit is not None:
            for key, value in parameters_fit.items():
                fit_ins.config[key] = value

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
        if server_round == 1:
            self.init_time = time.time()  # Initial time once first config has been done

            """analyst = DataAnalyst(
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
            export_thread.start()"""

        return fit_config, self.client_mapping

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes, str, int]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        counters: Dict,
        w_global: Optional[Parameters],
        prev_updates: Optional[Dict] = None,
        times: Dict[str, List[float]] = {},
        parameters_history: Dict = None,
        angles_history: Dict = None,
        prev_grad: np.ndarray = None,
        alpha: float=0.3,
        beta: float=0.25,
        k: int=5,
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
        # Call modified aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        g_global = parameters_to_ndarrays(w_global)

        if self.inplace:
            # Does in-place weighted average of results
            #aggregated_ndarrays = aggregate_inplace(results)
            results_trimmed = [(r[0], r[1]) for r in results]
            aggregated_ndarrays = aggregate_inplace(results_trimmed)
            aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        else:
            if prev_grad is not None:
                prev_grad_flat = _flatten_parameters(prev_grad)
            else:
                prev_grad_flat = None

            print(f"Params_hist keys: {list(parameters_history.keys())}")
            results_stale = []
            results_non_stale = []

            for client_proxy, fit_res, client_str, init_round in results:
                if self.prev_counters.get(client_str, 0) + 1 > 2:
                    results_stale.append((client_proxy, fit_res, client_str, init_round))
                else:
                    results_non_stale.append((client_proxy, fit_res, client_str))

            if results_non_stale:
                print(f"Non-stale: {len(results_non_stale)}, stale: {len(results_stale)}")              
                total_non_stale_examples = sum(fit_res.num_examples for _, fit_res, _ in results_non_stale)

                g_non_stale = (
                    sum(fit_res.num_examples * (_flatten_parameters(parameters_to_ndarrays(fit_res.parameters)) - _flatten_parameters(g_global))
                        for _, fit_res, _ in results_non_stale)
                    / total_non_stale_examples
                )

                proj_stale_sum = 0
                num_examples_stale = 0
                
                angles = []  # para registrar los ángulos

                for _, fit_res, client_id in results_non_stale:
                    g_client = _flatten_parameters(parameters_to_ndarrays(fit_res.parameters)) - _flatten_parameters(parameters_to_ndarrays(w_global))
                    
                    if prev_grad_flat is not None:
                        dot = np.dot(g_client, prev_grad_flat)
                        norm_k = np.linalg.norm(g_client)
                        norm_prev = np.linalg.norm(prev_grad_flat)
                        
                        if norm_k > 0 and norm_prev > 0:
                            cos_theta = np.clip(dot / (norm_k * norm_prev), -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_theta))
                        else:
                            angle_deg = None  # undefined if zero norm
                        print(f"Angle between prev_grad and client {client_id}: {angle_deg}")


                for _, fit_res, client_id, init_round in results_stale:
                    print(f"Init_round of {client_id} = {init_round}.")
                    g_init = _flatten_parameters(parameters_to_ndarrays(parameters_history[init_round]))
                    g_fit_res = _flatten_parameters(parameters_to_ndarrays(fit_res.parameters))
                
                    # Gradiente local del cliente stale
                    g_k = _flatten_parameters(parameters_to_ndarrays(fit_res.parameters)) - _flatten_parameters(parameters_to_ndarrays(parameters_history[init_round]))

                    if prev_grad_flat is not None:
                        dot = np.dot(g_k, prev_grad_flat)
                        norm_k = np.linalg.norm(g_k)
                        norm_prev = np.linalg.norm(prev_grad_flat)
                        
                        if norm_k > 0 and norm_prev > 0:
                            cos_theta = np.clip(dot / (norm_k * norm_prev), -1.0, 1.0)
                            angle_deg = np.degrees(np.arccos(cos_theta))
                        else:
                            angle_deg = None  # undefined if zero norm
                        print(f"Angle between prev_grad and client {client_id}: {angle_deg}")

                    """print("g_init values: ", g_init[:2])
                    print("g_fit_res: ", g_fit_res[:2])
                    print("g_k: ", g_k[:2])"""

                    # --- calcular ángulo entre g_k y g_non_stale ---
                    dot_angle = np.dot(g_k, g_non_stale)
                    norm_k = np.linalg.norm(g_k)
                    norm_global = np.linalg.norm(g_non_stale)

                    """print("\n--- Cliente stale ---")
                    print("g_k[:2] =", g_k[:2])
                    print("g_non_stale[:2] =", g_non_stale[:2])"""

                    if norm_k > 0 and norm_global > 0:
                        cos_theta = np.clip(dot_angle / (norm_k * norm_global), -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_theta))
                        angles.append(angle_deg)

                        if client_id not in angles_history:
                            angles_history[client_id] = []
                        angles_history[client_id].append(angle_deg)

                        # Penalizar aquellos ángulos mayores de 120
                        if angle_deg >= 120.0:
                            penalty = 0.0
                        else:
                            penalty = 1.0 - (angle_deg / 120.0)

                        """print(f"dot_angle = {dot_angle}")
                        print(f"angle_deg = {angle_deg:.4f}°")
                        print(f"penalty = {penalty:.4f}")"""
                    else:
                        penalty = 1.0
                        """print("Norma cero → penalty = 1.0")"""

                    age = server_round - init_round
                    age_factor = max(0, 1 - age / k)
                    penalty_final = penalty * age_factor

                    # Proyección de g_k sobre g_non_stale
                    norm_sq = np.dot(g_non_stale, g_non_stale)

                    g_proj_raw = (dot_angle / norm_sq) * g_non_stale
                    g_proj = (dot_angle / norm_sq) * g_non_stale * penalty_final

                    """print("norm_sq =", norm_sq)
                    print("g_proj_raw[:2] =", g_proj_raw[:2])
                    print("g_proj_penalized[:2] =", g_proj[:2])
                    print("penalty_final: ", penalty_final)"""

                    # Acumulamos ponderando por el número de ejemplos
                    proj_stale_sum += fit_res.num_examples * g_proj
                    num_examples_stale += fit_res.num_examples

                # Gradiente promedio de proyecciones stale
                if num_examples_stale > 0:
                    g_stale_proj = proj_stale_sum / num_examples_stale
                else:
                    g_stale_proj = np.zeros_like(g_non_stale)  # nada que añadir

                total_examples = total_non_stale_examples + num_examples_stale

                g_global_final = (
                    total_non_stale_examples * g_non_stale + num_examples_stale * g_stale_proj
                ) / total_examples
                
                #g_global_final = g_non_stale + g_stale_proj

                """print("g_global_final: ", g_global_final[:2])
                print("g_stale_proj: ", g_stale_proj[:2])"""

                g_global_structured = _unflatten_parameters(g_global_final, g_global)
                aggregated_ndarrays = [w + g for w, g in zip(g_global, g_global_structured)]
                aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

                w_old = parameters_to_ndarrays(w_global)
                w_new = parameters_to_ndarrays(aggregated_parameters)

                g_final = [new - old for new, old in zip(w_new, w_old)]

                """if angles:
                    print(f"Ángulos stale/global — media: {np.mean(angles):.2f}°, min: {np.min(angles):.2f}°, max: {np.max(angles):.2f}°")"""

                """total_bias = {
                    key1: sum(
                        abs(prev_updates[key1] - prev_updates[key2])
                        for key2, value2 in counters.items()
                        if value2 != 0 and key1 != key2
                    ) if value1 == 0 else 0
                    for key1, value1 in counters.items()
                }"""

                #print(f"Total bias aggregate_fit: {total_bias}")

                #mu_dict = {cid: np.exp(-beta * (counter + np.log1p(total_bias[cid] / k))) for cid, counter in counters.items()}
                #mu_dict = {cid: np.exp(-beta * counter) for cid, counter in counters.items()}
                # Convert results
                """weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples * mu_dict[cid_str])
                    for _, fit_res, cid_str in results
                ]

                aggregated_ndarrays = aggregate(weights_results)"""

                """weights_results = []
                for _, fit_res, cid_str in results:
                    local_params = parameters_to_ndarrays(fit_res.parameters)
                    delta_i = [lp - gp for lp, gp in zip(local_params, global_params)]
                    weight = fit_res.num_examples * mu_dict[cid_str]
                    weights_results.append((delta_i, weight))

                avg_delta = aggregate(weights_results)

                aggregated_ndarrays = [
                    gp + avg_d for gp, avg_d in zip(global_params, avg_delta)
                ]

                aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
            # Incorporar w_global si se proporciona
            if w_global is not None and alpha < 1.0:
                global_ndarrays = parameters_to_ndarrays(w_global)
                # Mezcla ponderada: alpha*new + (1-alpha)*global
                aggregated_ndarrays = [
                    alpha * new + (1 - alpha) * old
                    for new, old in zip(aggregated_ndarrays, global_ndarrays)
                ]"""
        
            else:
                aggregated_parameters = w_global

        self.prev_counters = counters

        # Aggregate custom metrics if aggregation fn was provided
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res, _, _ in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        
        # Extract the config parameters
        temp_config = self.config["tempConfig"]
        checkpoint_path = self.config["paths"]["checkpoint"]
        localcheckpoint_path = self.config["paths"]["localCheckpoint"]

        # Format the checkpoint path with the extracted values
        formatted_path = checkpoint_path.format(
            strategy=self.strategy_name,
            num_exec=temp_config["num_exec"],
            federation=temp_config["federation"],
            sub_execution=temp_config["execution_name"],
        )

        local_formatted_path = localcheckpoint_path.format(
            strategy=self.strategy_name,
            num_exec=temp_config["num_exec"],
            federation=temp_config["federation"],
            sub_execution=temp_config["execution_name"],
        )

        # Expand user directory and store the result
        directory_name = os.path.expanduser(formatted_path)
        localdirectory_name = os.path.expanduser(local_formatted_path)

        # Create the directory if it does not exist
        os.makedirs(directory_name, exist_ok=True)
        os.makedirs(localdirectory_name, exist_ok=True)

        # Write client-specific metrics to self.data from fit
        for client, fit_res, _, _ in results:
            id = self.client_mapping[client.cid]

            if self.server_type == 1:
                self.timestamps[id]["fit"] = str(times[id][-1])
            elif self.server_type == 2:
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

                saved_files = [
                    f for f in os.listdir(directory_name) if re.match(r"round-\d+-weights\.npz", f)
                ]

                def extract_round_number(filename):
                    match = re.search(r"round-(\d+)-weights\.npz", filename)
                    return int(match.group(1)) if match else float('inf')

                saved_files.sort(key=extract_round_number)

                if len(saved_files) >= 5:
                    oldest_file = saved_files[0]
                    os.remove(os.path.join(directory_name, oldest_file))

                # Save the aggregated arrays to the file
                self.filename = f"{directory_name}/round-{server_round + self.round_offset}-weights.npz"
                np.savez(self.filename, *aggregated_ndarrays)

            # Write server-specific metrics to self.data
            for metric in self.metrics:
                self.data["server"]["fit"][metric] = aggregated_metrics[metric]

            # Record the timestamp for the server's fit process
            self.timestamps["server"]["fit"] = time.time() - self.init_time

        # Guardar métricas de fit en logs CSV
        self._write_logs(localdirectory_name)

        return aggregated_parameters, aggregated_metrics, angles_history, g_final

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

            # Guardar métricas de evaluate en logs CSV
            self._write_logs(directory_name)

        return loss_aggregated, metrics_aggregated


    def _write_logs(self, directory_name: str):
        """Flatten data and timestamps and write them into CSV logs."""
        flattened_logs = {}        # Diccionario para aplanar métricas
        flattened_timestamps = {}  # Diccionario para aplanar tiempos

        # Aplana todas las métricas y timestamps recolectados
        for node in self.nodes:
            for epoch in self.epochs:
                flattened_timestamps[f"{node}_{epoch}"] = self.timestamps[node][epoch]
                for metric in self.metrics:
                    flattened_logs[f"{node}_{epoch}_{metric}"] = self.data[node][epoch][metric]

        # Abrir ficheros de log
        with open(f"{directory_name}/log_{self.num_exec}.csv", mode="a", newline="") as file_logs, \
            open(f"{directory_name}/timestamp_{self.num_exec}.csv", mode="a", newline="") as file_timestamps:

            writer_csv_logs = csv.writer(file_logs)
            writer_csv_timestamps = csv.writer(file_timestamps)

            # Escribir cabeceras si el fichero está vacío
            if file_logs.tell() == 0:
                writer_csv_logs.writerow(list(flattened_logs.keys()))
            if file_timestamps.tell() == 0:
                writer_csv_timestamps.writerow(list(flattened_timestamps.keys()))

            # Escribir valores
            writer_csv_logs.writerow(list(flattened_logs.values()))
            writer_csv_timestamps.writerow(list(flattened_timestamps.values()))


def _flatten_parameters(param_list):
    return np.concatenate([p.flatten() for p in param_list])

def _unflatten_parameters(flat_vector, template_list):
    arrays = []
    idx = 0
    for arr in template_list:
        size = arr.size
        arrays.append(flat_vector[idx:idx+size].reshape(arr.shape))
        idx += size
    return arrays
