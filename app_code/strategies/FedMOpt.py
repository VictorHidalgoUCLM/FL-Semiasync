from typing import Any, Dict, Optional
import math
import numpy as np
from logging import INFO
from flwr.common.logger import log

from flwr.common import (
    FitIns,
    Parameters,
)

from collections import defaultdict

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import os
import re

from app_code.strategies.FedAvg import FedAvgCustom
from app_code.strategies.early_stopping import EarlyStoppingTriggered, EMAEarlyStopping

class FedMOpt(FedAvgCustom):
    def __init__(
        self,
        *args: Any,
        **kwargs: Optional[Dict[str, Any]]
    ):
        """Initialize the FedMOpt strategy.
        
        Args:
            *args: Additional positional arguments for the superclass.
            **kwargs: Additional keyword arguments for the superclass.
        """

        super().__init__(*args, **kwargs)
        self.m = len(self.config["names"])  # First training round is syncrhonous
        self.data_history = {}  # Keep data history of last round
        self.init_time_round = 0

        self.utilities = []
        self.times = []

        self.early_stopper = EMAEarlyStopping(alpha=0.3, patience=3, min_delta=0.01)    # EMAEarlyStopping class object
        self.best_parameters = None     # Best checkpoint before early_stopping

    def aggregate_fit(self, server_round, results, failures, counters, times):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures, counters, times)
        
        # Check if model was improved, if it was, save it for later
        if self.early_stopper.improved:
            self.best_parameters = aggregated_parameters
        
        return aggregated_parameters, aggregated_metrics


    def aggregate_evaluate(self, server_round, results, failures, times):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures, times)
        
        # Update early_stopper only if server_round > 1, if early_stopping is detected...
        if server_round > 1 and not self.early_stopper.end_flag and self.early_stopper.update(self.data['server']['ev']['loss']):
            self.early_stopper.restart()    # Restart the early_stopper for the last evaluation
            
            log(INFO, f"Early stopping detected, round {server_round}")
            log(INFO, "")

            # Decomment for normal use, else early stopping won't happen
            """# Save the best checkpoint with the maximum rounds permitted so run.py knows it ended
            # because of early_stopping and not because of an error
            dir_name, filename = os.path.split(self.filename)
            match = re.match(r"round-(\d+)-weights\.npz", filename)
            if not match:
                raise ValueError(f"Nombre de archivo no válido: {filename}")

            config = self.config["config"]            

            new_filename = f"round-{int(config['rounds'])}-weights.npz"
            new_path = os.path.join(dir_name, new_filename)
            os.rename(self.filename, new_path)

            # Raise an excection that will lead to the last evaluation of checkpoint
            raise EarlyStoppingTriggered(f"Early stopping en server_round {server_round}")
            """
        return loss_aggregated, metrics_aggregated


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        fit_config = super().configure_fit(server_round, parameters, client_manager)

        if server_round > 1:   
            estadisticas = _get_statistics(self.timestamps, self.data_history, self.m, self.init_time_round)
            self.init_time_round = self.timestamps['server']['ev']

            # Initialize data for every function
            f1 = {}
            f2 = {}
            f3 = {}
            class_count = defaultdict(int)

            max_data_quantity = -math.inf
            max_unique_labels = -math.inf
            
            # Preprocess all info
            for node, values in self.prop.items():
                quantity = values.properties["quantity_data"]
                labels = list(map(int, values.properties["labels"].split(",")))                
                
                # Update max_data_quantity
                if quantity > max_data_quantity:
                    max_data_quantity = quantity

                # Update max unique labels
                unique_labels_len = len(labels)
                if unique_labels_len > max_unique_labels:
                    max_unique_labels = unique_labels_len

                # Count classes
                for label in labels:
                    class_count[label] += 1

                # Calculate f2 and f3 for every node
                f2[node] = values.properties["quantity_data"] / estadisticas[node]['media_fit']
                f3[node] = estadisticas[node]['loss_on_fit']

            # Complete for f1
            f1['q_max'] = max_data_quantity
            f1['c_max'] = max_unique_labels
            f1['unique'] = class_count
            
            # Scale for f3            
            f3['scale'] = 10

            # Calculate utility for every node
            for node, values in self.prop.items():
                utility, _ = _calculate_utility(f1=f1, f2=f2, f3=f3, node=node, values=values)
                estadisticas[node]["utility"] = utility

            # Calculate optimal m and prepare next round
            self.m, self.utilities, self.times = _calculate_optimal_m(100, estadisticas)        
            self.data_history = {}          
        return fit_config

def _calculate_optimal_m(simulations: int, estadisticas: Dict, alpha: float = 0.7):
    best_m = None
    best_score = -math.inf

    timestamp_list = [v['media_fit'] for k, v in estadisticas.items() if k != 'server']
    variability_list = [v['desviacion_fit'] for k, v in estadisticas.items() if k != 'server']
    utility_list = [v['utility'] for k, v in estadisticas.items() if k != 'server']    

    utilities = []
    times = []

    for m in range(1, len(utility_list) + 1):
        acc_utility = 0
        acc_time = 0

        for _ in range(simulations):
            simulated_times = [
                max(0.1, np.random.normal(mu, sigma))  # evitamos tiempos negativos
                for mu, sigma in zip(timestamp_list, variability_list)
            ]

            timers_simulation, execution_counters = _simulate_semiasynchronous(simulated_times, m)
            estimated_utility, estimated_time = _process_utility(simulated_times, timers_simulation, execution_counters, utility_list, m)

            acc_utility += estimated_utility
            acc_time += estimated_time

        mean_utility = acc_utility / simulations
        mean_time = acc_time / simulations        

        utilities.append(float(mean_utility))
        times.append(mean_time)

    max_utility = max(utilities)
    min_time = min(times)

    for idx, (mean_utility, mean_time) in enumerate(zip(utilities, times)):
        utility_score = mean_utility / max_utility if max_utility > 0 else 0.0
        time_score = min_time / mean_time if mean_time > 0 else 0.0

        score = alpha * utility_score + (1 - alpha) * time_score

        print(f"Score: {score}, utility_score: {utility_score}, time_score: {time_score}.")

        if score > best_score:
            best_score = score
            best_m = idx + 1

    log(INFO, f"Best score was M = {best_m}")
    log(INFO, "")
    return best_m, utilities, times

def _simulate_semiasynchronous(timestamp_list, m):
    timers_simulation = timestamp_list.copy()
    execution_counters = [0] * len(timestamp_list)
    stop = False

    def m_minimos_con_indices(lista, m):
        # Emparejar cada valor con su índice
        enumerada = list(enumerate(lista))
        
        # Ordenar por valor
        ordenada = sorted(enumerada, key=lambda x: x[1])
        
        # Tomar los m primeros (los menores)
        m_menores = ordenada[:m]
        
        # Separar valores e índices
        indices = [i for i, _ in m_menores]
        valores = [v for _, v in m_menores]
        
        return valores, indices
    
    while(True):
        valores, indices = m_minimos_con_indices(timers_simulation, m)
        max_value = max(valores)

        for value, index in zip(valores, indices):
            execution_counters[index] = execution_counters[index] + 1

            if all(counter != 0 for counter in execution_counters):
                for i, val in enumerate(execution_counters):
                    if i != index:
                        execution_counters[i] = val + 1
                stop = True
                break

            else:
                timers_simulation[index] = value + (max_value - value) + timestamp_list[index]

        if stop:
            break

    return timers_simulation, execution_counters

def _process_utility(timestamp_list, timers_simulation, execution_counters, utility_vector, m):
    total_times = [a * b for a, b in zip(timestamp_list, execution_counters)]
    max_time = max(timers_simulation)
    estimated_utility = 0

    for i in range(len(timers_simulation)):
        time_training = total_times[i] / max_time
        estimated_utility += time_training * utility_vector[i] * 0.95**(execution_counters[i])

    # Penalización por M pequeño
    alpha = 0.25
    penalizacion_m = 1 - alpha * (1 / m)
    estimated_utility *= penalizacion_m
    return estimated_utility, max(total_times)

def _get_statistics(timestamps, data_history, m, start_time_round=0):
    n_clientes = len(timestamps) - 1      # para quitar últimos n tiempos globales

    # 1. Extraer todos los tiempos de fit en lista global
    all_fit_times = []
    for node, vals in timestamps.items():
        fit_val = vals['fit']
        if isinstance(fit_val, str):
            all_fit_times.extend([float(x) for x in fit_val.split(';')])
        else:
            all_fit_times.append(float(fit_val))

    # 2. Ordenar y calcular mid_fits
    all_fit_times_sorted = sorted(all_fit_times)[:-n_clientes]
    mid_fits = all_fit_times_sorted[m-1::m]

    # 3. Construir intervalos globales con mid_fits
    intervalos_globales = []
    prev = start_time_round
    for mid in mid_fits:
        intervalos_globales.append((prev, mid))
        prev = mid
    intervalos_globales.append((prev, float('inf')))  # último intervalo

    # 4. Asignar intervalos ajustados a cada supernode
    supernodes_fit = {}
    for node, vals in timestamps.items():
        fit_val = vals['fit']
        if isinstance(fit_val, str):
            fit_times = [float(x) for x in fit_val.split(';')]
        else:
            fit_times = [float(fit_val)]

        nodo_intervals = []
        start = start_time_round

        for fit_time in fit_times:
            for (int_start, int_end) in intervalos_globales:
                if int_start <= fit_time <= int_end:
                    nodo_intervals.append((start, fit_time))
                    start = int_end  # inicio siguiente intervalo es límite derecho del intervalo global
                    break

        supernodes_fit[node] = nodo_intervals

    # 5. Calcular duraciones de cada intervalo
    duraciones_fit = {}
    for node, intervals in supernodes_fit.items():
        duraciones_fit[node] = [end - start for start, end in intervals]

    # 6. Calcular media y desviación estándar
    medias = {}
    desviaciones = {}
    for node, diffs in duraciones_fit.items():
        arr = np.array(diffs)
        medias[node] = np.mean(arr)
        desviaciones[node] = np.std(arr)

    # 7. Crear diccionario con estadísticas por supernode
    estadisticas = {}
    for node in timestamps.keys():
        if node == 'server':
            continue

        diff_sum = sum(data_history[node])
        estadisticas[node] = {
            'media_fit': medias[node],
            'desviacion_fit': desviaciones[node],
            'loss_on_fit': diff_sum,
        }

    return estadisticas

def _calculate_utility(f1, f2, f3, node, values,
    pesos={"w1": 0.3, "w2": 0.5, "w3": 0.2},
    pesos_f1={"r": 0.3, "c": 0.5, "u": 0.2}
):
    # f1: Calidad de datos
    labels = list(map(int, values.properties['labels'].split(',')))
    c_ratio = len(labels)
    q_ratio = values.properties["quantity_data"]
    
    def client_rarity_value_normalized(classes_in_client, client_class_count, total_clients):
        raw_value = 0.0
        for c in classes_in_client:
            rarity = 1.0 - (client_class_count.get(c, 0) / total_clients)
            raw_value += rarity

        if not classes_in_client:
            return 0.0  # evitar división por 0

        max_possible = len(classes_in_client) * (1.0 - (1.0 / total_clients))
        normalized_value = raw_value / max_possible if max_possible > 0 else 0.0
        return normalized_value

    u_ratio = client_rarity_value_normalized(labels, f1['unique'], 8)

    f1_score = (
        pesos_f1["r"] * c_ratio / f1['c_max'] +
        pesos_f1["c"] * q_ratio / f1['q_max'] +
        pesos_f1["u"] * u_ratio
    )
    
    # Obtener todos los valores
    data_seconds_values = list(f2.values())
    min_data_seconds = min(data_seconds_values)
    max_data_seconds = max(data_seconds_values)

    # Calcular f2: mayor velocidad = mejor puntuación
    f2_score = (f2[node] - min_data_seconds) / (max_data_seconds - min_data_seconds)

    # f3: Aporte de loss
    f3_score = np.tanh(f3['scale'] * f3[node])
    
    #print(f"Node {node}: f1_score {f1_score}, f2_score {f2_score}, f3_score {f3_score}.")
    
    # Utilidad total
    utilidad = (
            pesos["w1"] * f1_score +
            pesos["w2"] * f2_score +
            pesos["w3"] * f3_score
        )
    
    return round(utilidad, 4), {"f1": f1_score, "f2": f2_score, "f3": f3_score}
