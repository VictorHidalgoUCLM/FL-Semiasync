import requests
import toml
import os
import time
import csv
from logging import INFO, WARNING
from flwr.common.logger import log


class DataAnalyst:
    def __init__(
        self, prometheus_url: str, num_exec: int, strategy_name: str  # API for querying prometheus
    ):
        """Initialize the DataAnalyst class with necessary parameters and configurations.

        Args:
            prometheus_url (str): URL for Prometheus API.
            num_exec (int): Execution number for tracking.
            strategy_name (str): Name of the strategy being used.
        """
        # Variable initialization, saves every parameter on self. variable
        self.init_time = 0
        self.prometheus_url = prometheus_url
        self.num_exec = num_exec
        self.strategy_name = strategy_name
        self.toml_config = toml.load(os.environ.get("CONFIG_PATH"))

        # Load configuration variables and paths from the TOML file
        self.configVariable = self.toml_config["tempConfig"]
        self.configPath = self.toml_config["paths"]
        self.ruta_metrics = self.configPath["metrics"].format(
            federation=self.configVariable["federation"],
            strategy=self.strategy_name,
            sub_execution=self.configVariable["execution_name"],
        )

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
        """Scrapes Prometheus YAML config file locally, searches for
        hostnames, and saves them in self.devices and self.names.
        """

        self.devices = self.toml_config[self.configVariable["federation"]]["devices"]
        self.names = {
            key: value
            for key, value in self.toml_config["names"].items()
            if key != "clientapps"
        }
        self.clientapps = self.toml_config["names"].get("clientapps", [])

    def create_queries(self):
        """Creates Prometheus queries for both recursive and one-time execution
        based on the federation type and device configurations.
        """
        
        recursive_queries = [
            'container_network_transmit_bytes_total{{hostname="{device}",name="{name}"}}',
            'container_network_receive_bytes_total{{hostname="{device}",name="{name}"}}',
            'container_cpu_usage_seconds_total{{hostname="{device}",name="{name}"}}',
            'container_memory_usage_bytes{{hostname="{device}",name="{name}"}}',
            'node_memory_SwapFree_bytes{{hostname="{device}"}}',
            'node_thermal_zone_temp{{hostname="{device}"}}',
            'container_memory_failures_total{{hostname="{device}",name="{name}",scope="container"}}',
        ]

        one_time_queries = [
            'machine_cpu_cores{{hostname="{device}"}}',
            'machine_memory_bytes{{hostname="{device}"}}',
            'node_memory_SwapFree_bytes{{hostname="{device}"}}',
        ]

        # Local execution has only 1 device and several containers
        if self.configVariable["federation"] == "local-execution":
            # Devices are the supernodes
            devices_list = list(self.names.keys())

            # Create query for each supernode and client container
            for name in devices_list + self.clientapps:
                queries = [query.format(device=self.devices[0], name=name)
                        for query in recursive_queries]
                
                self.queries_recursive.append(queries)  # Store recursive query config 

                queries = [query.format(device=self.devices[0])
                        for query in one_time_queries]
                
                self.queries_one_time.append(queries)  # Store one_time query config

        # Remote execution has several devices and 2 containers per device
        elif self.configVariable["federation"] == "remote-execution":
            # Devices are each remote node
            devices_list = self.devices
            
            # Create client and supernode query for each device
            for i, device in enumerate(self.devices):
                queries = [query.format(device=device, name=list(self.names.keys())[i])
                        for query in recursive_queries]
                
                self.queries_recursive.append(queries)

            for i, device in enumerate(self.devices):
                queries = [query.format(device=device, name=self.clientapps[i])
                        for query in recursive_queries]

                self.queries_recursive.append(queries)  # Store recursive query config 

                queries = [query.format(device=device) for query in one_time_queries]
                self.queries_one_time.append(queries)  # Store one_time query config 

        for name in devices_list:
            # Create directory for storing metrics results for each hostname
            os.makedirs(f"{self.ruta_metrics}/{name}", exist_ok=True)

            output_file = f"{self.ruta_metrics}/{name}/ex_{self.num_exec}.csv"
            self.output_files.append(output_file)

    def execute_query(self, queries):
        """Executes a list of Prometheus queries and returns the results.

        Args:
            queries (list): List of Prometheus queries to execute.

        Returns:
            list: List of query results.
        """
        result = []

        for query in queries:
            response = requests.get(self.prometheus_api_url, params={"query": query})

            if response.status_code == 200:
                result.append(response.json()["data"]["result"])
            else:
                print(
                    f"Error executing query. Status code: {response.status_code}"
                )
                return

        return result

    def clients_up(self):
        """Checks if all client devices are up and connected to Prometheus."""
        log(INFO, "Checking client connections...")
        query = 'up{{hostname="{device}"}}'

        for device in self.devices:
            while True:
                response = requests.get(
                    self.prometheus_api_url,
                    params={"query": query.format(device=device)},
                )

                if response.status_code == 200:
                    data = response.json()
                    result = data.get("data", {}).get("result", [])

                    if result:
                        if result[0]["value"][1] == "1":
                            log(INFO, f"Device {device} connected.")
                            break
                    else:
                        log(WARNING, f"Device {device} still not connected, waiting...")

                time.sleep(5)

    def execute_one_time_queries(self):
        """Executes all one-time queries and stores the results."""
        for query in self.queries_one_time:
            if self.configVariable["federation"] == "local-execution":
                self.result_one_time.append([None, None, None])
            else:
                query_result = self.execute_query(query)

                while any(not res for res in query_result):
                    query_result = self.execute_query(query)
                    print("No response received, waiting...")
                    time.sleep(5)

                query_values = [result[0]["value"][1] for result in query_result]
                self.result_one_time.append(query_values)

    def execute_recursive_queries(self):
        """
        Executes all recursive queries and appends the results to self.result_recursive.
        """
        self.elapsed_time = int(time.time() - self.init_time)
        self.result_recursive = []

        for query in self.queries_recursive:
            query_result = self.execute_query(query)

            query_values = [
                value["value"][1] for result in query_result for value in result
            ]
            self.result_recursive.append(query_values)

    def export_data(self):
        """
        Processes and exports the query results to CSV files.
        """
        if self.configVariable["federation"] == "local-execution":
            device_list = list(self.names.keys())
        elif self.configVariable["federation"] == "remote-execution":
            device_list = self.devices

        for i, _ in enumerate(device_list):
            output_file = self.output_files[i]

            cpu_total = self.result_one_time[i][0]
            memory_total = self.result_one_time[i][1]
            swap_total = self.result_one_time[i][2]

            result_supernode = self.result_recursive[i]
            result_clientapp = self.result_recursive[i + len(self.names)]

            fin_result = [
                (
                    str(float(result_supernode[j]) + float(result_clientapp[j]))
                    if j in [2, 3, 6, 7]
                    else (
                        str(result_clientapp[j])
                        if j in [0, 1]
                        else str(result_supernode[j])
                    )
                )
                for j in range(len(result_supernode))
            ]

            with open(output_file, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)

                if os.path.getsize(output_file) == 0:
                    header_row = [
                        "Time_stamp(s)",
                        "Net_transmit(B)",
                        "Net_receive(B)",
                        "CPU_time(s)",
                        "RAM_usage(B)",
                        "Swap_free(B)",
                        "Temp(C)",
                        "Mem_fault",
                        "Mem_majfault",
                        "",
                        cpu_total if cpu_total is None else int(cpu_total),
                        memory_total if memory_total is None else int(memory_total),
                        swap_total if swap_total is None else int(swap_total),
                    ]
                    csv_writer.writerow(header_row)

                row = [self.elapsed_time] + [float(value) for value in fin_result[:8]]
                csv_writer.writerow(row)
