import math
import os
import re
from collections import defaultdict
from logging import INFO
from itertools import product

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import FitIns, Parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

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
        self.init_time_round = [0]

        self.early_stopper = EMAEarlyStopping(
            alpha=0.3, patience=3, min_delta=0.01
        )    # EMAEarlyStopping class object

        self.best_parameters = None     # Best checkpoint before early_stopping

        self.duraciones_fit = {}
        self.prev_time = 0
        self.prev_updates = {}
        self.prev_prev_updates = {}

        self.beta = self.config["tempConfig"].get("beta", 0.75)
        self.k_param = self.config["tempConfig"].get("k", 7)
        self.slope = self.config["tempConfig"].get("slope", 0.03)

    def aggregate_fit(
            self,
            server_round,
            results,
            failures,
            counters,
            w_global,
            times,
        ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
            counters=counters,
            w_global=w_global,
            times=times,
            prev_updates=self.prev_updates,
            beta=self.beta,
            k=self.k_param,
        )

        prev_prev_time = self.prev_time
        self.prev_time = max(times[id][-1] for id in times)
        self.new_time = self.prev_time - prev_prev_time

        for key, counter in counters.items():
            self.prev_updates.setdefault(key, 0)
            self.prev_prev_updates[key] = self.prev_updates[key]
            if counter == 0:
                self.prev_updates[key] = server_round
        
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
            r"""# Save the best checkpoint with the maximum rounds permitted so run.py knows it ended 
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
            estadisticas = _get_statistics_semiasynchronous(
                server_round,
                self.timestamps,
                self.duraciones_fit,
                self.init_time_round,
                self.prev_updates,
                self.prev_prev_updates
            )

            self.init_time_round.append(self.timestamps['server']['ev'])

            for node, values in self.prop.items():
                # Save quantity data on client statistics
                estadisticas[node]['quantity_data'] = values.properties["quantity_data"]

            # Initialize data for every function
            f1 = {}
            class_count = defaultdict(int)

            max_data_quantity = -math.inf
            max_unique_labels = -math.inf
            
            # Preprocess all info
            for node, values in self.prop.items():
                quantity = values.properties["quantity_data"]
                labels = list(map(int, values.properties["labels"].split(",")))                
                
                # Update max values
                if quantity > max_data_quantity:
                    max_data_quantity = quantity
                if len(labels) > max_unique_labels:
                    max_unique_labels = len(labels)

                # Count classes
                for label in labels:
                    class_count[label] += 1

            # Complete for f1
            f1['q_max'] = max_data_quantity
            f1['c_max'] = max_unique_labels
            f1['unique'] = class_count

            f1_raw_dict = {}

            for node, values in self.prop.items():
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

                u_ratio = client_rarity_value_normalized(labels, class_count, 8)

                f1_raw = (
                    (0.4 * c_ratio) / f1['c_max'] +
                    (0.5 * q_ratio) / f1['q_max'] +
                     0.1 * u_ratio
                )

                f1_raw_dict[node] = f1_raw

            f1['raw'] = f1_raw_dict
            f1['mean'] = np.mean(list(f1_raw_dict.values()))
            f1['std'] = np.std(list(f1_raw_dict.values()))

            # Calculate optimal m and prepare next round
            self.m = _calculate_optimal_m(
                server_round=server_round,
                simulations=1,
                estadisticas=estadisticas,
                prev_counters=self.prev_counters,
                prev_max=self.new_time,
                prev_updates=self.prev_updates,
                k_param=self.k_param,
                beta=self.beta,
                slope=self.slope,
                f1=f1
            )           

        return fit_config

def _calculate_optimal_m(
        server_round: int,
        simulations: int,
        estadisticas: Dict[str, Dict[str, float]],
        prev_counters: List[int],
        prev_max: float,
        prev_updates: List[int],
        k_param: int,
        beta: float,
        slope: float,
        f1: Dict[str, Any],
    ) -> Tuple[int, List[float], List[float]]:

    mu_dict = {k: v['mu'] for k, v in estadisticas.items() if k != 'server'}
    sigma_dict = {k: v['sigma'] for k, v in estadisticas.items() if k != 'server'}

    M_candidates = list(range(1, len(mu_dict) + 1))
    best_score = -math.inf

    utilities = {m: [] for m in M_candidates}
    times = {m: [] for m in M_candidates}
    scores = []

    for _ in range(simulations):
        simulated_times = {}
        f2 = {}
        for k in mu_dict:
            simulated_time = np.random.lognormal(mean=mu_dict[k], sigma=sigma_dict[k])
            simulated_times[k] = simulated_time
            f2[k] = estadisticas[k]['quantity_data'] / simulated_time

        utility_dict = {
            node: _calculate_utility(f1=f1, f2=f2, node=node)
            for node in f2
        }

        for m in M_candidates:
            est_utility, est_time =_simulate_semiasynchronous(
                server_round=server_round,
                simulated_times=simulated_times,
                utility_dict=utility_dict,
                prev_counters=prev_counters,
                prev_max=prev_max,
                prev_updates=prev_updates,
                m=m,
                beta=beta,
                k=k_param)
            
            utilities[m].append(est_utility)
            times[m].append(est_time)  

    med_utilities = {m: np.median(u_list) for m, u_list in utilities.items()}
    med_times = {m: np.median(t_list) for m, t_list in times.items()}

    max_utility = max(med_utilities.values())
    min_time = min((t for t in med_times.values() if t > 0), default=0)

    for m in M_candidates:
        med_utility = med_utilities[m]
        med_time = med_times[m]

        utility_score = max(0.0, min(med_utility / max_utility, 1.0)) if max_utility > 0 else 1.0
        time_score = max(0.0, min(min_time / med_time, 1.0)) if med_time > 0 else 1.0

        score = 2 * utility_score * time_score / (utility_score + time_score) if (utility_score + time_score) > 0 else 0.0
        scores.append(score)

        print(f"Score: {score}, utility_score: {utility_score}, time_score: {time_score}.")

        if score >= best_score:
            best_score = score
            best_m = m

    for k in range(best_m, len(M_candidates)):
        if scores[k] >= (best_score * (1 - slope)):
            best_m = k+1
            log(INFO, f"Best score updated, M={k+1}")

    log(INFO, f"Best score was M = {best_m}")
    log(INFO, "")
    return best_m


def _simulate_semiasynchronous(
        server_round: int,
        simulated_times: List[float],
        utility_dict: Dict[int, float],
        prev_counters: Dict[int, int],
        prev_max: float,
        prev_updates: Dict[int, int],
        m: int,
        beta: float = 0.75,
        k: int = 7,
    ):
    execution_counters = prev_counters.copy()
    updated_times = simulated_times.copy()
    
    total_utility = 0

    for cid, counter_val in execution_counters.items():
        if counter_val != 0:
            updated_times[cid] = max(0, simulated_times[cid] - prev_max)

    if m == 1:
        print("--- Simulation Step ---")
        print(f"  Simulated times: {simulated_times}")
        print(f"  Prev counters: {execution_counters}")
        print(f"  Utility dict: {utility_dict}")
        print(f"  Prev max: {prev_max}")
        print(f"  Prev updates: {prev_updates}")
        print(f"  Adjusted simulated times: {updated_times}")

    print(f"--- M = {m} ---")

    def m_minimos_con_indices(diccionario, m):
        ordenada = sorted(diccionario.items(), key=lambda x: x[1])        
        m_menores = ordenada[:m]
        indices = [i for i, _ in m_menores]
        valores = [v for _, v in m_menores]
        return valores, indices
    
    valores, indices = m_minimos_con_indices(updated_times, m)
    max_value = max(valores) if valores else 0

    print(f"  Chosen indices: {indices} with max time {max_value}")

    # --- Counter updates ---
    for cid in execution_counters.keys():
        if cid in indices:  # clients that updated last round
            execution_counters[cid] = 0
        else:
            execution_counters[cid] += 1

    print(f"  Updated execution counters: {execution_counters}")

    for cid, counter_val in execution_counters.items():
        if counter_val == 0:
            total_bias = sum(
                server_round - prev_updates[cid2]
                for cid2, cval2 in execution_counters.items()
                if cval2 != 0 and cid2 != cid
            )
        else:
            total_bias = 0

        """if cid in indices or updated_times[cid] <= 0:
            t_i = 1.0
        else:
            t_i = max_value / updated_times[cid]"""
        
        t_i = max_value / updated_times[cid] if updated_times[cid] > 0 else 0.0

        mu_i = np.exp(-beta * (counter_val + np.log1p(total_bias / k)))
        total_utility += utility_dict[cid] * mu_i * t_i

        print(f"    Utility contribution from client {cid}: {utility_dict[cid]} * {mu_i} * {t_i} = {utility_dict[cid] * mu_i * t_i}")
    return total_utility, max_value


def _get_statistics_semiasynchronous(
        server_round,
        timestamps,
        duraciones_fit,
        start_time_round,
        prev_updates,
        prev_prev_updates
    ):
    # 1. Construir supernodes_fit y actualizar duraciones_fit en un solo paso
    for node, vals in timestamps.items():
        if node == 'server' or prev_updates[node] != (server_round - 1):
            continue

        if prev_prev_updates[node] == (server_round - 2):
            init_round = server_round - 2
        else:
            init_round = prev_prev_updates[node]

        # Duración no negativa
        fit_time = float(vals['fit'])
        dur = fit_time - start_time_round[init_round]
        if dur > 0:
            duraciones_fit.setdefault(node, []).append(dur)

    # 2. Calcular media y sigma logarítmicas
    weights = np.array(2 ** np.arange(3))
    estadisticas = {}

    for node in duraciones_fit.keys():
        # 1. Tomar los últimos N tiempos y aplicar log
        times_window = duraciones_fit[node][-len(weights):]
        if len(times_window) == 0:
            continue

        y = np.log(times_window)       
        w = weights[-len(y):]  # recortar pesos si hay menos elementos

        # 2. Media y desviación ponderadas en log-escala
        mu = np.sum(w * y) / np.sum(w)
        sigma = np.sqrt(np.sum(w * (y - mu)**2) / np.sum(w))

        estadisticas[node] = {
            'mu': mu,
            'sigma': sigma, 
        }
    return estadisticas


def _calculate_utility(f1, f2, node,
    weights={"w1": 0.0, "w2": 1.0, "w3": 0.0}
):  
    f1_value = f1['raw'][node]
    f1_mean = f1['mean']
    f1_std = f1['std']
    
    # Calcular f1: mayor iidness = mejor puntuación
    f1_score = (f1_value - f1_mean) / (3 * f1_std) if f1_std > 0 else 1.0
    f1_score = max(0.0, min(f1_score, 1.0))

    # Obtener todos los valores
    times = np.array(list(f2.values()))
    ratio = times.max() / times.min()

    f2_values = np.log(times) if ratio > 1.5 else times
    f2_mean = np.mean(f2_values)
    f2_std = np.std(f2_values)

    value_i = np.log(f2[node]) if ratio > 1.5 else f2[node]

    # Calcular f2: mayor velocidad = mejor puntuación
    if f2_std > 0:
        z = (f2_mean - value_i) / (3 * f2_std)
        f2_score = 1 - z
    else:
        f2_score = 0.0
    f2_score = np.clip(f2_score, 0.0, 1.0)
    
    # Utilidad total
    utility = (
            weights["w1"] * f1_score +
            weights["w2"] * f2_score
        )
    
    #print(f"Utility calculation for node {node}: f1={f1_score}, f2={f2_score}, total={utility}")
    return round(utility, 4)
