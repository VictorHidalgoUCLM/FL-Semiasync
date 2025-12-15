import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import toml
import os
import re

def lighten_color(color, factor=0.7):
    # Convert to a numpy array to work with
    color = np.array(color)
    # Blend with white: increase brightness by a factor
    return tuple(color + (1 - color) * (1 - factor))

def darken_color(color, factor=0.7):
    # Convert to a numpy array to work with
    color = np.array(color)
    # Blend with black: decrease brightness by a factor
    return tuple(color * factor)

projectconf = '../projectconf.toml'
data_types = ['iid']
strategies = ['FedAvg']
final_strat = 'FedMOpt'
federation = 'local-execution'
available = 8
num_execs = 1

root = f'../{federation}/results'

acc_columns = ['loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5', 'loss_6', 'loss_7', 'loss_8', 'loss_9']
time_columns = ['timestamp_1', 'timestamp_2', 'timestamp_3', 'timestamp_4', 'timestamp_5', 'timestamp_6', 'timestamp_7', 'timestamp_8', 'timestamp_9']

acc_columns = acc_columns[:available + 1]
time_columns = time_columns[:available + 1]

og_colors = plt.cm.tab10.colors

y_min_global = np.inf
y_max_global = -np.inf

def find_numeric_dirs(base_path, prefix):
    """Devuelve lista de valores numéricos en directorios tipo prefixN o prefixN.M."""
    result = []
    pattern = re.compile(rf"{prefix}(\d+(\.\d+)?)$")

    if not os.path.isdir(base_path):
        return result

    for name in os.listdir(base_path):
        full = os.path.join(base_path, name)
        if os.path.isdir(full):
            m = pattern.match(name)
            if m:
                value = m.group(1)
                # INT si no tiene punto, FLOAT si lo tiene
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
                result.append(value)

    return result


def interpolate_at_time(timestamps, values, target_time):
    """
    Retorna el valor interpolado de 'values' en el tiempo 'target_time'.
    Maneja todos los casos extremos de forma robusta.
    
    timestamps: array-like
    values: array-like
    target_time: float
    """

    # Convertimos a numpy
    t = np.array(timestamps, dtype=float)
    v = np.array(values, dtype=float)

    # Ordenar por timestamp, por si vienen desordenados
    order = np.argsort(t)
    t = t[order]
    v = v[order]

    # Eliminar NaN (puede ocurrir en merges o CSVs incompletos)
    mask = ~np.isnan(t) & ~np.isnan(v)
    t = t[mask]
    v = v[mask]

    # Si no hay datos, retornar NaN
    if len(t) == 0:
        return np.nan

    # Caso exacto: target_time coincide con un timestamp
    exact = np.where(t == target_time)[0]
    if len(exact) > 0:
        return v[exact[0]]

    # Si target_time es menor que el mínimo → devolvemos v[0]
    if target_time < t[0]:
        return v[0]

    # Si target_time es mayor que el máximo → devolvemos el último valor
    if target_time > t[-1]:
        return v[-1]

    # Interpolación segura
    return np.interp(target_time, t, v)


config = toml.load(projectconf)

# Detectar parámetros directamente desde el filesystem
slows = set()
betas = {}  # betas[slow] = [beta1, beta2, ...]
ks = {}     # ks[(slow, beta)] = [k1, k2, ...]

for data_type in data_types:
    root_path = os.path.join(root, final_strat, data_type)

    # Detectar slows (ordenados)
    found_slows = sorted(find_numeric_dirs(root_path, "slow"))
    for slow in found_slows:
        slows.add(slow)

        slow_dir = os.path.join(root_path, f"slow{slow}")

        # Detectar betas dentro de slowX
        found_betas = sorted(find_numeric_dirs(slow_dir, "beta"))
        betas[slow] = found_betas

        for beta in found_betas:
            beta_dir = os.path.join(slow_dir, f"beta{beta}")

            # Detectar ks dentro de betaX
            found_ks = sorted(find_numeric_dirs(beta_dir, "k"))
            ks[(slow, beta)] = found_ks

results_list = []
time_objective = []
for strategy in strategies:
    for data_type in data_types:
        for slow in sorted(slows):
            for beta in sorted(betas.get(slow, [])):
                for k in sorted(ks.get((slow, beta), [])):
                    merged_df = None
                    for sync_client in range(1, available + 1):
                        log_path = config['paths']['localLog'].format(federation=federation, strategy=strategy, sub_execution=f"{data_type}/sync{sync_client}/slow{slow}", num_exec=1)
                        time_path = config['paths']['localTimestamp'].format(federation=federation, strategy=strategy, sub_execution=f"{data_type}/sync{sync_client}/slow{slow}", num_exec=1)

                        log_path = f"../{log_path}"
                        time_path = f"../{time_path}"

                        temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                        temp_time_df = pd.read_csv(time_path)[["server_ev"]]

                        final_df = pd.concat([temp_log_df, temp_time_df], axis=1)
                        final_df.columns = [f'accuracy_{sync_client}', f'loss_{sync_client}', f'timestamp_{sync_client}']

                        if merged_df is None:
                            merged_df = final_df
                        else:
                            merged_df = pd.merge(merged_df, final_df, how="outer", left_index=True, right_index=True)

                    exec_metrics = []
                    exec_times = []
                    for num_exec in range(num_execs):
                        log_path = config['paths']['localLog'].format(federation=federation, strategy=final_strat, sub_execution=f"{data_type}/slow{slow}/beta{beta}/k{k}", num_exec=1)
                        time_path = config['paths']['localTimestamp'].format(federation=federation, strategy=final_strat, sub_execution=f"{data_type}/slow{slow}/beta{beta}/k{k}", num_exec=1)

                        log_path = f"../{log_path}"
                        time_path = f"../{time_path}"

                        metric = pd.read_csv(log_path)[["server_ev_loss"]].to_numpy().ravel()
                        time = pd.read_csv(time_path)[["server_ev"]].to_numpy().ravel()

                        exec_metrics.append(metric)
                        exec_times.append(time)
                    
                    t_max = min(time[-1] for time in exec_times)
                    t_grid = np.linspace(0, t_max, 50)

                    all_interp = []
                    for time, metric in zip(exec_times, exec_metrics):
                        interp = np.interp(t_grid, time, metric)
                        all_interp.append(interp)

                    acc_mean = np.mean(all_interp, axis=0)

                    final_df = pd.DataFrame({
                        f'loss_{available+1}': acc_mean,
                        f'timestamp_{available+1}': t_grid
                    })
                    merged_df = pd.merge(merged_df, final_df, how="outer", left_index=True, right_index=True)

                    for i in range(1, available + 2):
                        # Calculamos el mínimo y máximo de loss para establecer el rango global
                        y_min_global = min(y_min_global, merged_df[f"loss_{i}"].min())
                        y_max_global = max(y_max_global, merged_df[f"loss_{i}"].max())

                    last_valids = merged_df[time_columns].apply(lambda col: col.dropna().iloc[-1])
                    max_time = last_valids.min()
                    time_objective.append(max_time)

                    estimations = []
                    for timestamp_col, loss_col in zip(time_columns, acc_columns):
                        timestamps = merged_df[timestamp_col].dropna().values
                        values = merged_df[loss_col].dropna().values
                        estimation = interpolate_at_time(timestamps, values, max_time)
                        estimations.append(estimation)
                    
                    simplified_df = pd.Series(estimations, index=acc_columns)
                    simplified_df[acc_columns] = simplified_df[acc_columns].div(simplified_df[f'loss_{available}'], axis=0)
                    simplified_df[acc_columns] = simplified_df[acc_columns] - 1
                    simplified_df[acc_columns] = simplified_df[acc_columns] * 100
                    simplified_df[acc_columns] = simplified_df[acc_columns].round(2)
                    simplified_df = simplified_df.drop(f'loss_{available + 1}')

                    results_list.append(simplified_df)

                    fig, axs = plt.subplots(1, 1, figsize=(10, 6))  # 'sharex=True' para compartir el eje x (timestamp)

                    for i in range(1, available + 2):
                        if i == available + 1:
                            axs.plot(merged_df[f"timestamp_{i}"], merged_df[f"loss_{i}"], marker='o', markersize=4, linestyle='-', label=f"{final_strat}", color=og_colors[i-1])
                        else:
                            axs.plot(merged_df[f"timestamp_{i}"], merged_df[f"loss_{i}"], marker='o', markersize=4, linestyle='-', label=f"M = {i}", color=og_colors[i-1])

                    # Establece los límites del eje Y usando el rango global calculado
                    axs.set_ylim(y_min_global - 0.1, 3)  # Un margen pequeño para mayor claridad
                    axs.set_xlim(0, 4000)

                    axs.set_xlabel("Timestamp (s)", fontsize=18)
                    axs.set_ylabel("Loss", fontsize=18)
                    axs.axvline(x=max_time, color='black', linestyle='--')
                    axs.legend(fontsize=16)
                    axs.grid(True)

                    axs.tick_params(axis='both', which='major', labelsize=16)
                    axs.set_title(f"Comparative Loss Graph - {data_type.upper()} - Slow Client: {slow}\nBeta: {beta}, K: {k}", fontsize=20)

                    plt.tight_layout()

                    os.makedirs(f'../{federation}/results/{final_strat}/{data_type}/graphs', exist_ok=True)
                    plt.savefig(f'../{federation}/results/{final_strat}/{data_type}/graphs/slow{slow}_beta{beta}_k{k}.png', dpi=150)

                    loss_cols = [col for col in merged_df.columns if "loss" in col]
                    
                    #print(f"Execution {slow}")
                    for loss_col in loss_cols:
                        # índice de la fila donde ocurre el mínimo de la columna loss
                        idx_min = merged_df[loss_col].idxmin()
                        
                        # nombre de la columna timestamp correspondiente
                        n = loss_col.split("_")[1]         # extrae el número
                        timestamp_col = f"timestamp_{n}"
                        
                        # valor del timestamp en esa fila
                        ts_value = merged_df.at[idx_min, timestamp_col]
                        min_loss = merged_df.at[idx_min, loss_col]

                        #print(f"{loss_col}: min = {merged_df.at[idx_min, loss_col]} en timestamp = {ts_value}.")

            results = {}  # key = (beta, k) → value = dataframe
            for beta in sorted(betas.get(slow, [])):
                for k in sorted(ks.get((slow, beta), [])):
                    exec_metrics = []
                    exec_times = []

                    sub_exec = f"{data_type}/slow{slow}/beta{beta}/k{k}"

                    for exec_id in range(num_execs):

                        log_path = (
                            "../" +
                            config['paths']['localLog'].format(
                                federation=federation,
                                strategy=f'{final_strat}',
                                sub_execution=sub_exec,
                                num_exec=exec_id + 1
                            )
                        )

                        time_path = (
                            "../" +
                            config['paths']['localTimestamp'].format(
                                federation=federation,
                                strategy=f'{final_strat}',
                                sub_execution=sub_exec,
                                num_exec=exec_id + 1
                            )
                        )

                        # Leer datos
                        log_df = pd.read_csv(log_path)[["server_ev_loss"]]
                        time_df = pd.read_csv(time_path)[["server_ev"]]

                        metric = log_df.to_numpy().ravel()
                        time = time_df.to_numpy().ravel()

                        exec_metrics.append(metric)
                        exec_times.append(time)

                    # Interpolamos en una misma malla temporal
                    t_max = min(t[-1] for t in exec_times)
                    t_grid = np.linspace(0, t_max, 50)

                    interpolated_runs = [
                        np.interp(t_grid, t, m)
                        for t, m in zip(exec_times, exec_metrics)
                    ]

                    mean_loss = np.mean(interpolated_runs, axis=0)

                    # Guardar
                    results[(beta, k)] = pd.DataFrame({
                        "timestamp": t_grid,
                        "loss": mean_loss
                    })

            # =======================================================
            # 2. GRAFICAS FIJANDO beta
            # =======================================================
            for beta in sorted(betas.get(slow, [])):
                ks_for_beta = sorted(ks.get((slow, beta), []))
                if len(ks_for_beta) <= 1:
                    continue  # solo hay un k → no graficar

                fig, ax = plt.subplots(figsize=(10, 6))

                for idx, k in enumerate(ks_for_beta):
                    df = results[(beta, k)]
                    color = og_colors[idx % len(og_colors)]

                    ax.plot(
                        df["timestamp"],
                        df["loss"],
                        marker="o",
                        markersize=4,
                        linestyle="-",
                        color=color,
                        label=f"k={k}"
                    )

                ax.set_title(f"{final_strat} — slow={slow}, data={data_type}, β={beta}", fontsize=18)
                ax.set_xlabel("Timestamp (s)", fontsize=18)
                ax.set_ylabel("Loss", fontsize=18)
                ax.legend(fontsize=14)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=14)
                plt.ylim(1.3, 2.8)
                plt.tight_layout()

                save_path = f'../{federation}/results/{final_strat}/{data_type}/graphs/beta_{beta}_slow{slow}.png'
                plt.savefig(save_path, dpi=300)
                plt.close()

            # =======================================================
            # 2. GRAFICAS FIJANDO K
            # =======================================================
            for k in sorted(ks.get((slow, beta), [])):
                betas_for_k = [
                    beta for beta in sorted(betas.get(slow, []))
                    if k in ks.get((slow, beta), [])
                ]

                if len(betas_for_k) <= 1:
                    continue  # solo hay un beta → no graficar

                fig, ax = plt.subplots(figsize=(10, 6))

                for idx, beta in enumerate(betas_for_k):
                    df = results[(beta, k)]
                    color = og_colors[idx % len(og_colors)]

                    ax.plot(
                        df["timestamp"],
                        df["loss"],
                        marker="o",
                        markersize=4,
                        linestyle="-",
                        color=color,
                        label=f"β={beta}"
                    )

                ax.set_title(f"{final_strat} — slow={slow}, data={data_type}, k={k}", fontsize=18)
                ax.set_xlabel("Timestamp (s)", fontsize=18)
                ax.set_ylabel("Loss", fontsize=18)
                ax.legend(fontsize=14)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=14)
                plt.ylim(1.3, 2.8)
                plt.tight_layout()

                save_path = f'../{federation}/results/{final_strat}/{data_type}/graphs/k_{k}_slow{slow}.png'
                plt.savefig(save_path, dpi=300)
                plt.close()
