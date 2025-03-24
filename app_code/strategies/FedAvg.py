from flwr.server.strategy import FedAvg
from logging import INFO
from flwr.common.logger import log
import os
import threading
import numpy as np
import flwr as fl
import time
import toml 
import csv
from concurrent.futures import ThreadPoolExecutor

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    GetPropertiesIns,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from app_code.strategies.data_analyst import DataAnalyst
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

class ExportThread(threading.Thread):
    def __init__(self, analyst_instance, sleep_time, init_time):
        super().__init__()
        self.analyst_instance = analyst_instance
        self.sleep_time = sleep_time
        self.init_time = init_time

    def run(self):
        try:
            self.analyst_instance.init_time = self.init_time
            while True:
                self.analyst_instance.execute_recursive_queries()
                self.analyst_instance.export_data()
                time.sleep(self.sleep_time)

        except IndexError as e:
            print("Error en Run")
            

class FedAvgCustom(FedAvg):
    def __init__(self, num_exec, strategy_name, debug, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.init_time = time.time()
        self.client_mapping = {}
        self.debug = debug
    
        self.round_offset = 0
        self.config = toml.load(os.environ.get('CONFIG_PATH'))

        self.prometheus_conf = self.config.get('prometheus_conf', {})
        self.federation_conf = self.config[self.config['tempConfig']['federation']]

        nodes = list(self.config['names'].keys())
        self.nodes = ['server'] + [dispositivo for dispositivo in nodes if dispositivo != "clientapps"]

        self.epochs = ['fit', 'ev']
        self.metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1']
        self.data = {}
        self.timestamps = {}

        for node in self.nodes:
            for epoch in self.epochs:
                if node not in self.timestamps:
                    self.timestamps[node] = {}
                if epoch not in self.timestamps[node]:
                    self.timestamps[node][epoch] = {}

                for metric in self.metrics:
                    if node not in self.data:
                        self.data[node] = {}
                    if epoch not in self.data[node]:
                        self.data[node][epoch] = {}
                    self.data[node][epoch][metric] = 0


    def set_round_offset(self, offset):
        self.round_offset = offset
    
    def process_client(self, client, fit_ins, parameters_fit, server_round):
        if server_round == 1:
            prop = client.get_properties(GetPropertiesIns({}), timeout=None, group_id=None)
            if prop.properties["user"]:
                    id = prop.properties["user"]
                    self.client_mapping[client.cid] = id

        id = self.client_mapping[client.cid]
        client_conf = self.config['names'].get(id)
        fit_ins.config = {
                'epochs': client_conf[0],
                'batch_size': client_conf[1],
                'subset_size': client_conf[2],
                'server_round': server_round,
                'debug': self.debug,
            }

        if parameters_fit is not None:
            for key, value in parameters_fit.items():
                fit_ins.config[key] = value

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_fit: Optional[dict] = None    
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Get client/config standard pairs from the FedAvg superclass
        fit_config = super().configure_fit(
            server_round, parameters, client_manager
        )

        with ThreadPoolExecutor() as executor:
            for client, fit_ins in fit_config:
                executor.submit(self.process_client, client, fit_ins, parameters_fit, server_round)

        # Initialize data analyst if it is the first round
        if server_round == 1:
            analyst = DataAnalyst(self.prometheus_conf['prometheus_url'], self.num_exec, self.strategy_name)

            # Get hostnames and create queries
            analyst.get_hostnames()
            analyst.create_queries()
            analyst.clients_up()

            # Execute one-time queries
            analyst.execute_one_time_queries()

            # Start data export thread
            export_thread = ExportThread(analyst, self.prometheus_conf['sleep_time'], self.init_time)
            export_thread.daemon = True
            export_thread.start()
        return fit_config
        

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        times : Dict[str, List[float]] = {}
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        directory_name = os.path.expanduser(self.config['paths']['checkpoint'].format(strategy=self.strategy_name, num_exec=self.config['tempConfig']['num_exec'], federation=self.config['tempConfig']['federation'], sub_execution=self.config['tempConfig']['execution_name']))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        # Writing supernode metrics at self.data
        for client, fit_res in results:
            id = self.client_mapping[client.cid]
            self.timestamps[id]['fit'] = ';'.join(map(str, times[id]))

            for metric in self.metrics:
                self.data[id]['fit'][metric] = fit_res.metrics[metric]
                
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)

            if server_round % 5 == 0 or server_round + self.round_offset == self.config['config']['rounds']:
                # Save aggregated_ndarrays every 5 rounds
                log(INFO, f"")
                log(INFO, f"Saving round {server_round} aggregated_ndarrays...")
                log(INFO, f"")

                np.savez(f"{directory_name}/round-{server_round+self.round_offset}-weights.npz", *aggregated_ndarrays)

            for metric in self.metrics:
                self.data['server']['fit'][metric] = aggregated_metrics[metric]

            self.timestamps['server']['fit'] = time.time() - self.init_time

        return aggregated_parameters, aggregated_metrics
    

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        times: Optional[Dict[str, float]] = {}
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        directory_name = os.path.expanduser(self.config['paths']['logs'].format(strategy=self.strategy_name, federation=self.config['tempConfig']['federation'], sub_execution=self.config['tempConfig']['execution_name']))

        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        for client, evaluate_res in results:
            id = self.client_mapping[client.cid]
            self.timestamps[id]['ev'] = times[id]

            for metric in self.metrics:
                if metric == 'loss':
                    self.data[id]['ev'][metric] = evaluate_res.loss
                else:
                    self.data[id]['ev'][metric] = evaluate_res.metrics[metric]


        if loss_aggregated is not None and metrics_aggregated is not None:

            for metric in self.metrics:
                if metric == 'loss':
                    self.data['server']['ev'][metric] = loss_aggregated
                else:
                    self.data['server']['ev'][metric] = metrics_aggregated[metric]

            self.timestamps['server']['ev'] = time.time() - self.init_time

            flattened_data = {}
            flattened_timestamps = {}

            for node in self.nodes:
                for epoch in self.epochs:
                    flattened_timestamps[f'{node}_{epoch}'] = self.timestamps[node][epoch]
                    for metric in self.metrics:
                        flattened_data[f'{node}_{epoch}_{metric}'] = self.data[node][epoch][metric]

            with open(f"{directory_name}/log_{self.num_exec}.csv", mode='a', newline='') as file:
                escritor_csv = csv.writer(file)

                if file.tell() == 0:
                    escritor_csv.writerow(list(flattened_data.keys()))

                escritor_csv.writerow(list(flattened_data.values()))

            with open(f"{directory_name}/timestamp_{self.num_exec}.csv", mode='a', newline='') as file:
                escritor_csv = csv.writer(file)

                if file.tell() == 0:
                    escritor_csv.writerow(list(flattened_timestamps.keys()))

                escritor_csv.writerow(list(flattened_timestamps.values()))

        return loss_aggregated, metrics_aggregated