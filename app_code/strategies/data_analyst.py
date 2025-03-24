"""
Data_Analyst will be instanciated as an object at a custom strategy,
creates queries for each client and metric, which will be executed every
5 seconds while training is done.

This class scrapes prometheus in search of performance metrics, stored
at a Pandas DataFrame.
"""

# Necessary modules
import requests
import toml
import os
import time
import csv
from logging import INFO, WARNING
from flwr.common.logger import log

# DataAnalyst class, used as object at custom strategies
class DataAnalyst:
    def __init__(
            self,
            prometheus_url,             # API for querying prometheus
            num_exec,
            strategy_name):                    # Config file
        
        # Variable initialization, saves every parameter on self. variable
        self.init_time = 0
        self.prometheus_url = prometheus_url
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.toml_config = toml.load(os.environ.get('CONFIG_PATH'))
        
        self.configVariable = self.toml_config['tempConfig']
        self.configPath = self.toml_config['paths']
        self.ruta_metrics = self.configPath['metrics'].format(federation=self.configVariable['federation'], strategy=self.strategy_name, sub_execution=self.configVariable['execution_name'])
        
        # Other variables
        self.elapsed_time = 0
        self.prometheus_api_url = f"{self.prometheus_url}/api/v1/query"
        self.devices = []
        self.names = {}
        self.output_files = []
        self.queries_one_time = []
        self.queries_recursive = []
        self.result_one_time = []
        self.result_recursive = []
        self.clientapps = []


    def get_hostnames(self):
        """
        Scrapes prometheus yaml config file locally, searches for
        hostnames and saves them at self.raspberry_pis.
        """
        self.devices = self.toml_config[self.configVariable['federation']]['devices']
        self.names = {key: value for key, value in self.toml_config['names'].items() if key != "clientapps"}
        self.clientapps = self.toml_config['names'].get("clientapps", [])


    def create_queries(self):
        recursive_queries = [
            'container_network_transmit_bytes_total{{hostname="{device}",name="{name}"}}',
            'container_network_receive_bytes_total{{hostname="{device}",name="{name}"}}',
            'container_cpu_usage_seconds_total{{hostname="{device}",name="{name}"}}',
            'container_memory_usage_bytes{{hostname="{device}",name="{name}"}}',
            'node_memory_SwapFree_bytes{{hostname="{device}"}}',
            'node_thermal_zone_temp{{hostname="{device}"}}',
            'container_memory_failures_total{{hostname="{device}",name="{name}",scope="container"}}'
            ]

        one_time_queries = [
            'machine_cpu_cores{{hostname="{device}"}}',
            'machine_memory_bytes{{hostname="{device}"}}',
            'node_memory_SwapFree_bytes{{hostname="{device}"}}'
            ]

        if self.configVariable['federation'] == "local-execution":
            disp_list = list(self.names.keys())

            for name in list(self.names.keys()) + self.clientapps:
                # For loop that stores all queries at self.queries_recursive and self.queries_one_time
                queries = [query.format(device=self.devices[0], name=name)
                        for query in recursive_queries]
                
                self.queries_recursive.append(queries)

                queries = [query.format(device=self.devices[0])
                        for query in one_time_queries]
                
                self.queries_one_time.append(queries)

        elif self.configVariable['federation'] == 'remote-execution':
            disp_list = self.devices
            print(disp_list)

            for i, device in enumerate(self.devices):
                # For loop that stores all queries at self.queries_recursive and self.queries_one_time
                queries = [query.format(device=device, name=list(self.names.keys())[i])
                        for query in recursive_queries]
                
                print(device)
                print(queries)
                
                self.queries_recursive.append(queries)

            for i, device in enumerate(self.devices):
                queries = [query.format(device=device, name=self.clientapps[i])
                        for query in recursive_queries]

                self.queries_recursive.append(queries)

                queries = [query.format(device=device) for query in one_time_queries]
                self.queries_one_time.append(queries)

        for name in disp_list:
            # Creates directory for storing metrics results for each hostname
            if not os.path.exists(f'{self.ruta_metrics}/{name}'):
                os.makedirs(f'{self.ruta_metrics}/{name}')

            output_file = f'{self.ruta_metrics}/{name}/ex_{self.num_exec}.csv'
            self.output_files.append(output_file)      


    def execute_query(self, queries):
        """
        Executes one query and returns result
        """
        result = []

        for query in queries:
            response = requests.get(
                self.prometheus_api_url, params={
                    'query': query})

            if response.status_code == 200:
                result.append(response.json()['data']['result'])
            else:
                print(f"Error al realizar la consulta. Código de estado: {response.status_code}")
                return

        return result


    def clients_up(self):
        log(INFO, "Checking client connections...")
        query = 'up{{hostname="{device}"}}'
        
        for device in self.devices:
            while True:
                response = requests.get(self.prometheus_api_url, params={'query': query.format(device=device)})

                if response.status_code == 200:
                    data = response.json()
                    result = data.get("data", {}).get("result", [])

                    if result:
                        if result[0]["value"][1] == '1':
                            log(INFO, f"Device {device} connected.")
                            log(INFO, f"")
                            break
                    
                    else:
                        log(WARNING, f"Device {device} still not connected, waiting...")

                time.sleep(5)
                           
    def execute_one_time_queries(self):
        """
        Executes all one_time_queries, calling execute_query
        If all queries are responded, the program can continue
        """
        for query in self.queries_one_time:
            if self.configVariable['federation'] == 'local-execution':
                self.result_one_time.append([None, None, None])
            
            else:
                query_result = self.execute_query(query)

                while any(not res for res in query_result):
                    query_result = self.execute_query(query)
                    print("No se recibe, esperando...")
                    time.sleep(5)

                query_values = [result[0]['value'][1] for result in query_result]
                self.result_one_time.append(query_values)
           

    def execute_recursive_queries(self):
        """
        Executes all recurive queries, appending results at self.result_recursive
        """
        self.elapsed_time = int(time.time() - self.init_time)
        self.result_recursive = []

        for query in self.queries_recursive:
            query_result = self.execute_query(query)

            query_values = [value['value'][1]
                            for result in query_result for value in result]
            
            self.result_recursive.append(query_values)


    def export_data(self):
        """
        Scrapes, cleans and stores all results of the queries
        """
        if self.configVariable['federation'] == "local-execution":
            device_list = list(self.names.keys())

        elif self.configVariable['federation'] == "remote-execution":
            device_list = self.devices

        for i, _ in enumerate(device_list):
            output_file = self.output_files[i]

            cpu_total = self.result_one_time[i][0]
            memory_total = self.result_one_time[i][1]
            swap_total = self.result_one_time[i][2]

            result_supernode = self.result_recursive[i]
            result_clientapp = self.result_recursive[i + len(self.names)]

            fin_result = [
                str(float(result_supernode[j]) + float(result_clientapp[j])) if j in [2, 3, 6, 7] 
                else str(result_clientapp[j]) if j in [0, 1] 
                else str(result_supernode[j])
                for j in range(len(result_supernode))
            ]

            with open(output_file, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                if os.path.getsize(output_file) == 0:
                    header_row = [
                        'Time_stamp(s)',
                        'Net_transmit(B)',
                        'Net_receive(B)',
                        'CPU_time(s)',
                        'RAM_usage(B)',
                        'Swap_free(B)',
                        'Temp(C)',
                        'Mem_fault',
                        'Mem_majfault',
                        '',
                        cpu_total if cpu_total is None else int(cpu_total),
                        memory_total if memory_total is None else int(memory_total),
                        swap_total if swap_total is None else int(swap_total)
                        ]
                    
                    csv_writer.writerow(header_row)

                row = [self.elapsed_time] + [float(value) for value in fin_result[:8]]
                csv_writer.writerow(row)
