import math
from collections import defaultdict
from logging import INFO

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import FitIns, Parameters
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from app_code.strategies.FedAvg import FedAvgCustom

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
        self.m = 8  # First training round is syncrhonous
        self.init_time_round = [0]

        self.prev_time = 0
        self.new_time = 0
        self.holt_state = {}
        self.estadisticas = {}

        self.slope = self.config["tempConfig"].get("slope", 0.00)
        self.debug = self.config["tempConfig"].get("debug", False)

    def aggregate_fit(
            self,
            server_round,
            results,
            failures,
            counters,
            times,
            global_parameters,
            prev_updates,
            beta,
            gamma
        ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round=server_round,
            results=results,
            failures=failures,
            counters=counters,
            global_parameters=global_parameters,
            prev_updates=prev_updates,
            beta=beta,
            gamma=gamma,
        )

        if server_round > 0:
            prev_prev_time = self.prev_time

            valid_ids = [self.client_mapping[client.cid] for client, *_ in results]
            self.prev_time = max(times[cid][-1] for cid in valid_ids)

            self.new_time = self.prev_time - prev_prev_time
        return aggregated_parameters, aggregated_metrics


    def aggregate_evaluate(self, server_round, results, failures, times):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures, times)
        return loss_aggregated, metrics_aggregated


    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        prev_updates: List[int],
        beta: float = 0.9,
        gamma: float = 0.9
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""       
        fit_config = super().configure_fit(server_round, parameters, client_manager)   

        if server_round > self.init_round:  
            self.init_time_round.append(self.timestamps['server']['ev'])

            for node, values in self.prop.items():
                self.estadisticas.setdefault(node, {})['quantity_data'] = values.properties["quantity_data"]

            simulations = 1000 if not self.debug else 1
            self.estadisticas, self.holt_state = _get_statistics_semiasynchronous(
                server_round,
                self.timestamps,
                self.init_time_round,
                prev_updates,
                self.holt_state,
                self.estadisticas,
                self.init_rounds
            )

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
                simulations=simulations,
                estadisticas=self.estadisticas,
                prev_counters=self.prev_counters,
                prev_max=self.new_time,
                prev_updates=prev_updates,
                beta=beta,
                slope=self.slope,
                f1=f1,
                debug=self.debug,
            )           

        return fit_config

def _calculate_optimal_m(
        simulations: int,
        estadisticas: Dict[str, Dict[str, float]],
        prev_counters: List[int],
        prev_max: float,
        prev_updates: List[int],
        beta: float,
        slope: float,
        f1: Dict[str, Any],
        debug: bool = False,
    ) -> Tuple[int, List[float], List[float]]:

    mu_dict = {k: v['mu'] for k, v in estadisticas.items() if k != 'server'}
    sigma_dict = {k: v['sigma'] for k, v in estadisticas.items() if k != 'server'}

    M_candidates = list(range(1, len(mu_dict) + 1))
    best_score = -math.inf

    velocities = {m: [] for m in M_candidates}
    times = {m: [] for m in M_candidates}
    scores = []

    for _ in range(simulations):
        simulated_times = {}
        velocity_dict = {}
        for k in mu_dict:
            simulated_time = np.random.lognormal(mean=mu_dict[k], sigma=sigma_dict[k])
            simulated_times[k] = simulated_time
            velocity_dict[k] = estadisticas[k]['quantity_data'] / simulated_time # Data per time unit

        for m in M_candidates:
            est_velocity, est_time =_simulate_semiasynchronous(
                simulated_times=simulated_times,
                velocity_dict=velocity_dict,
                prev_counters=prev_counters,
                prev_max=prev_max,
                prev_updates=prev_updates,
                m=m,
                beta=beta,
                debug=debug,
            )
            
            velocities[m].append(est_velocity)
            times[m].append(est_time)  

    med_velocities = {m: np.median(v_list) for m, v_list in velocities.items()}
    med_times = {m: np.median(t_list) for m, t_list in times.items()}

    vel_p10 = np.percentile(
        np.array(list(med_velocities.values()), dtype=float),
        10
    )

    time_p10 = np.percentile(
        np.array(list(med_times.values()), dtype=float),
        10
    )

    max_velocity = max(med_velocities.values())
    min_time = min((t for t in med_times.values() if t > 0), default=0)

    for m in M_candidates:
        med_velocity = med_velocities[m]
        med_time = med_times[m]

        velocity_score_1 = max(0.0, min(med_velocity / max_velocity, 1.0)) if max_velocity > 0 else 1.0
        time_score_1 = max(0.0, min(min_time / med_time, 1.0)) if med_time > 0 else 1.0

        velocity_score_2 = max(0.0, min(med_velocity / vel_p10, 1.0)) if vel_p10 > 0 else 1.0
        time_score_2 = max(0.0, min(time_p10 / med_time , 1.0)) if med_time > 0 else 1.0

        alpha = 0.4
        score_1 = velocity_score_1 * time_score_1 / (alpha * velocity_score_1 + (1 - alpha) * time_score_1) if (velocity_score_1 + time_score_1) > 0 else 0.0
        score_2 = velocity_score_2 * time_score_2 / (alpha * velocity_score_2 + (1 - alpha) * time_score_2) if (velocity_score_2 + time_score_2) > 0 else 0.0
        
        scores.append(score_1)

        log(INFO, f"Score 1: {score_1}, velocity_score: {velocity_score_1}, time_score: {time_score_1}.")
        #log(INFO, f"Score 2: {score_2}, velocity_score: {velocity_score_2}, time_score: {time_score_2}.")

        if score_1 >= best_score:
            best_score = score_1
            best_m = m

    for k in range(best_m, len(M_candidates)):
        if scores[k] >= (best_score * (1 - slope)):
            best_m = k+1
            log(INFO, f"Best score updated, M={k+1}")

    log(INFO, f"Best score was M = {best_m}")
    log(INFO, "")
    return best_m


def _simulate_semiasynchronous(
        simulated_times: List[float],
        velocity_dict: Dict[int, float],
        prev_counters: Dict[int, int],
        prev_max: float,
        prev_updates: Dict[int, int],
        m: int,
        beta: float = 0.9,
        debug: bool = False,
    ):
    execution_counters = prev_counters.copy()
    updated_times = simulated_times.copy()
    
    total_velocity = 0

    for cid, counter_val in execution_counters.items():
        if counter_val != 0:
            updated_times[cid] = max(0, simulated_times[cid] - prev_max)

    if m == 1 and debug:
        print("--- Simulation Step ---")
        print(f"  Simulated times: {simulated_times}")
        print(f"  Prev counters: {execution_counters}")
        print(f"  Velocity dict: {velocity_dict}")
        print(f"  Prev max: {prev_max}")
        print(f"  Prev updates: {prev_updates}")
        print(f"  Adjusted simulated times: {updated_times}")

    if debug:
        print(f"--- M = {m} ---")

    def m_minimos_con_indices(diccionario, m):
        ordenada = sorted(diccionario.items(), key=lambda x: x[1])        
        m_menores = ordenada[:m]
        indices = [i for i, _ in m_menores]
        valores = [v for _, v in m_menores]
        return valores, indices
    
    valores, indices = m_minimos_con_indices(updated_times, m)
    max_value = max(valores) if valores else 0

    if debug:
        print(f"  Chosen indices: {indices} with max time {max_value}")

    for cid, counter_val in execution_counters.items():
        """if counter_val == 0:
            total_bias = sum(
                server_round - prev_updates[cid2]
                for cid2, cval2 in execution_counters.items()
                if cval2 != 0 and cid2 != cid
            )
        else:
            total_bias = 0"""

        if cid in indices or updated_times[cid] <= 0:
            mu_i = beta ** counter_val
            n_i = 1/len(simulated_times)
            total_velocity += velocity_dict[cid] * mu_i * n_i
    return total_velocity, max_value


def _get_statistics_semiasynchronous(
        server_round,
        timestamps,
        start_time_round,
        prev_updates,
        holt_state,
        estadisticas,
        init_rounds: Dict = {}
    ):
    def update_holt(node, new_time, holt_state, n=3):       
            alpha = 2 / (n + 1)
            gamma = alpha / 3
            y = np.log(new_time)

            # Inicialización
            if node not in holt_state:
                holt_state[node] = {
                    "L": y,
                    "T": 0.0,
                    "sigma": 0.0
                }
                return holt_state

            L_prev = holt_state[node]["L"]
            T_prev = holt_state[node]["T"] 

            # Holt update
            L = alpha * y + (1 - alpha) * (L_prev + T_prev)
            T = gamma * (L - L_prev) + (1 - gamma) * T_prev

            # Actualizar sigma log (EMA de error)
            err = y - (L_prev + T_prev)
            beta_sigma = alpha / 2
            sigma = np.sqrt(
                beta_sigma * err**2 + (1 - beta_sigma) * holt_state[node]["sigma"]**2
            )

            holt_state[node]["L"] = L
            holt_state[node]["T"] = T
            holt_state[node]["sigma"] = sigma
            return holt_state

    for node, time in timestamps.items():
        if node == 'server' or prev_updates[node] != (server_round - 1):
            continue

        real_time = float(time['fit']) - start_time_round[init_rounds[node]]
        holt_state = update_holt(node, real_time, holt_state)

        estadisticas[node]['mu'] = holt_state[node]["L"] + holt_state[node]["T"]
        estadisticas[node]['sigma'] = holt_state[node]["sigma"]

    return estadisticas, holt_state


"""def _calculate_metrics(f1, f2, node,
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
    max_value = times.max()
    f2_score = f2[node] / max_value if max_value > 0 else 1.0

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
    return round(utility, 4)"""
