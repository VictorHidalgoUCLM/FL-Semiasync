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
data_types = ['noniid0.7', 'noniid0.5']
strategy = 'FedAvg'
final_strat = 'FedMOpt'
federation = 'local-execution'
available = 7
num_execs = 3
start = 5

root = f'../{federation}/results'

acc_columns = ['loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5', 'loss_6', 'loss_7', 'loss_8', 'loss_9']
time_columns = ['timestamp_1', 'timestamp_2', 'timestamp_3', 'timestamp_4', 'timestamp_5', 'timestamp_6', 'timestamp_7', 'timestamp_8', 'timestamp_9']

acc_columns = acc_columns[start:available + 1]
time_columns = time_columns[start:available + 1]

COLORS_BY_METHOD = {
    'sync5': '#1f77b4',    # Azul
    'sync6': '#ff7f0e',    # Naranja  
    'sync7': '#2ca02c',    # Verde
    'FedMOpt': '#d62728'   # Rojo
}
og_colors = plt.cm.tab10.colors

y_min_global = np.inf
y_max_global = -np.inf

def find_slow_dirs_univ(base_path):
    result = []
    pattern = re.compile(
        r"^slow(\d+(\.\d+)?)(?:_med(\d+(\.\d+)?))?$"
    )

    if not os.path.isdir(base_path):
        return result

    for name in os.listdir(base_path):
        full = os.path.join(base_path, name)
        if not os.path.isdir(full):
            continue

        m = pattern.match(name)
        if not m:
            continue

        slow = m.group(1)
        med = m.group(3)

        slow = float(slow) if "." in slow else int(slow)
        med = (
            float(med) if med and "." in med
            else int(med) if med
            else None
        )

        if med is None:
            result.append((slow,))
        else:
            result.append((slow, med))

    return result


def find_numeric_dirs(base_path, prefix):
    """Sin cambios: detecta valores numéricos en betaX o kY"""
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
                value = float(value) if "." in value else int(value)
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
slows = set()
betas = {}  # betas[slow_tuple] = [beta1, beta2, ...]
ks = {}     # ks[(slow_tuple, beta)] = [k1, k2, ...]

for data_type in data_types:
    root_path = os.path.join(root, final_strat, data_type)

    # Detectar slows como tuples (N, S)
    found_slows = sorted(find_slow_dirs_univ(root_path))
    for slow in found_slows:
        slows.add(slow)

        slow_dir = os.path.join(root_path, f"slow{slow[0]}")

        # Detectar betas dentro de slowNmedS
        found_betas = sorted(find_numeric_dirs(slow_dir, "beta"))
        betas[slow] = found_betas

        for beta in found_betas:
            beta_dir = os.path.join(slow_dir, f"beta{beta}")

            # Detectar ks dentro de betaX
            found_ks = sorted(find_numeric_dirs(beta_dir, "gamma"))
            ks[(slow, beta)] = found_ks

target_beta = 0.7
target_gamma = 12.0
results_list = []

for data_type in data_types:
    for slow in sorted(slows):

        slow_dir_name = f"slow{slow[0]}"

        # =========================
        # SOLO ESTE BETA
        # =========================
        if target_beta not in betas.get(slow, []):
            continue

        beta = target_beta
        gamma = target_gamma

        merged_df = None

        for sync_client in range(start, available + 1):
            temp_dfs = []

            for num_exec in range(1, num_execs + 1):
                log_path = config['paths']['localLog'].format(
                    federation=federation,
                    strategy=strategy,
                    sub_execution=f"{data_type}/sync{sync_client}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                    num_exec=num_exec
                )
                time_path = config['paths']['localTimestamp'].format(
                    federation=federation,
                    strategy=strategy,
                    sub_execution=f"{data_type}/sync{sync_client}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                    num_exec=num_exec
                )

                log_path = f"../{log_path}"
                time_path = f"../{time_path}"

                if os.path.exists(log_path):
                    temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                    temp_time_df = pd.read_csv(time_path)[["server_ev"]]
                else:
                    temp_log_df = pd.DataFrame(columns=["server_ev_accuracy", "server_ev_loss"])
                    temp_time_df = pd.DataFrame(columns=["server_ev"])

                temp_df = pd.concat([temp_log_df, temp_time_df], axis=1)
                temp_df.columns = [
                    f'server_ev_accuracy_{num_exec}',
                    f'server_ev_loss_{num_exec}',
                    f'server_ev_time_{num_exec}'
                ]
                temp_dfs.append(temp_df)

            df_all = pd.concat(temp_dfs, axis=1)

            accuracy_cols = [col for col in df_all.columns if "accuracy" in col]
            loss_cols = [col for col in df_all.columns if "loss" in col]
            timestamp_cols = [col for col in df_all.columns if "time" in col]

            final_df = pd.DataFrame({
                f'accuracy_{sync_client}': df_all[accuracy_cols].mean(axis=1),
                f'loss_{sync_client}': df_all[loss_cols].mean(axis=1),
                f'timestamp_{sync_client}': df_all[timestamp_cols].mean(axis=1),
            })

            if merged_df is None:
                merged_df = final_df
            else:
                merged_df = pd.merge(
                    merged_df, final_df,
                    how="outer",
                    left_index=True,
                    right_index=True
                )

        # =========================
        # FEDMOPT
        # =========================
        temp_dfs = []

        for num_exec in range(1, num_execs + 1):
            log_path = config['paths']['localLog'].format(
                federation=federation,
                strategy=final_strat,
                sub_execution=f"{data_type}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                num_exec=num_exec
            )
            time_path = config['paths']['localTimestamp'].format(
                federation=federation,
                strategy=final_strat,
                sub_execution=f"{data_type}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                num_exec=num_exec
            )

            log_path = f"../{log_path}"
            time_path = f"../{time_path}"

            if os.path.exists(log_path):
                temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                temp_time_df = pd.read_csv(time_path)[["server_ev"]]
            else:
                temp_log_df = pd.DataFrame(columns=["server_ev_accuracy", "server_ev_loss"])
                temp_time_df = pd.DataFrame(columns=["server_ev"])

            temp_df = pd.concat([temp_log_df, temp_time_df], axis=1)
            temp_df.columns = [
                f'server_ev_accuracy_{num_exec}',
                f'server_ev_loss_{num_exec}',
                f'server_ev_time_{num_exec}'
            ]
            temp_dfs.append(temp_df)

        df_all = pd.concat(temp_dfs, axis=1)

        accuracy_cols = [col for col in df_all.columns if "accuracy" in col]
        loss_cols = [col for col in df_all.columns if "loss" in col]
        timestamp_cols = [col for col in df_all.columns if "time" in col]

        final_df = pd.DataFrame({
            f'accuracy_{available + 1}': df_all[accuracy_cols].mean(axis=1),
            f'loss_{available + 1}': df_all[loss_cols].mean(axis=1),
            f'timestamp_{available + 1}': df_all[timestamp_cols].mean(axis=1),
        })

        merged_df = pd.merge(
            merged_df, final_df,
            how="outer",
            left_index=True,
            right_index=True
        )

        # =========================
        # RESTO DEL CÓDIGO IGUAL
        # =========================

        for i in range(start, available + 2):
            y_min_global = min(y_min_global, merged_df[f"loss_{i}"].min())
            y_max_global = max(y_max_global, merged_df[f"loss_{i}"].max())

        last_valids = merged_df[time_columns].apply(
            lambda col: col.dropna().iloc[-1] if col.notna().any() else 0.0
        )

        max_time = last_valids.min()

        estimations = []
        for timestamp_col, loss_col in zip(time_columns, acc_columns):
            timestamps = merged_df[timestamp_col].dropna().values
            values = merged_df[loss_col].dropna().values
            estimation = interpolate_at_time(timestamps, values, max_time)
            estimations.append(estimation)

        simplified_df = pd.Series(estimations, index=acc_columns)
        simplified_df[acc_columns] = simplified_df[acc_columns].div(
            simplified_df[f'loss_{available}'], axis=0
        )
        simplified_df[acc_columns] = simplified_df[acc_columns] - 1
        simplified_df[acc_columns] = simplified_df[acc_columns] * 100
        simplified_df[acc_columns] = simplified_df[acc_columns].round(2)

        simplified_df = simplified_df.drop(f'loss_{available + 1}')
        results_list.append(simplified_df)

        # =========================
        # PLOT
        # =========================
        fig, axs = plt.subplots(1, 1, figsize=(10, 6))

        for i in range(start, available + 2):
            if i == available + 1:
                # FedMOpt - SIEMPRE el mismo color
                color = COLORS_BY_METHOD['FedMOpt']
                label = "FedMDyn"
            else:
                # syncX - SIEMPRE el mismo color por M
                color = COLORS_BY_METHOD[f"sync{i}"]
                label = f"M = {i}"

            axs.plot(
                merged_df[f"timestamp_{i}"],
                merged_df[f"loss_{i}"],
                marker='o',
                markersize=4,
                linestyle='-',
                label=label,
                color=color
            )

        axs.set_ylim(y_min_global - 0.05, 3 + 0.05)
        axs.set_xlabel("Timestamp (s)", fontsize=18)
        axs.set_ylabel("Loss", fontsize=18)
        axs.legend(fontsize=16)
        axs.grid(True)
        axs.tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()

        save_path = f'../{federation}/results/{final_strat}/{data_type}/graphs/by_slow/{slow_dir_name}.pdf'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="pdf")

        plt.close()
"""# ==============================
# GRÁFICAS POR (slow, método) COMPARANDO BETAS
# ==============================

methods = list(range(start, available + 1)) + ["FedMOpt"]

for data_type in data_types:
    for slow in sorted(slows):

        slow_dir_name = f"slow{slow[0]}"
        slow_betas = sorted(betas.get(slow, []))

        for method in methods:

            fig, axs = plt.subplots(1, 1, figsize=(10, 6))

            y_min_global = np.inf
            y_max_global = -np.inf
            plotted_any = False

            for beta in slow_betas:
            #for gamma in ks.get((slow, beta), []):
                merged_df = None
                gamma = 12.0
                #beta = 0.7

                # ======================
                # CONSTRUCCIÓN merged_df
                # ======================
                for sync_client in range(start, available + 1):
                    temp_dfs = []

                    for num_exec in range(1, num_execs + 1):
                        log_path = config['paths']['localLog'].format(
                            federation=federation,
                            strategy=strategy,
                            sub_execution=f"{data_type}/sync{sync_client}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                            num_exec=num_exec
                        )
                        time_path = config['paths']['localTimestamp'].format(
                            federation=federation,
                            strategy=strategy,
                            sub_execution=f"{data_type}/sync{sync_client}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                            num_exec=num_exec
                        )

                        log_path = f"../{log_path}"
                        time_path = f"../{time_path}"

                        if os.path.exists(log_path):
                            temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                            temp_time_df = pd.read_csv(time_path)[["server_ev"]]
                        else:
                            temp_log_df = pd.DataFrame(columns=["server_ev_accuracy", "server_ev_loss"])
                            temp_time_df = pd.DataFrame(columns=["server_ev"])

                        temp_df = pd.concat([temp_log_df, temp_time_df], axis=1)
                        temp_df.columns = [
                            f'server_ev_accuracy_{num_exec}',
                            f'server_ev_loss_{num_exec}',
                            f'server_ev_time_{num_exec}'
                        ]
                        temp_dfs.append(temp_df)

                    df_all = pd.concat(temp_dfs, axis=1)

                    accuracy_cols = [col for col in df_all.columns if "accuracy" in col]
                    loss_cols = [col for col in df_all.columns if "loss" in col]
                    timestamp_cols = [col for col in df_all.columns if "time" in col]

                    final_df = pd.DataFrame({
                        f'accuracy_{sync_client}': df_all[accuracy_cols].mean(axis=1),
                        f'loss_{sync_client}': df_all[loss_cols].mean(axis=1),
                        f'timestamp_{sync_client}': df_all[timestamp_cols].mean(axis=1),
                    })

                    if merged_df is None:
                        merged_df = final_df
                    else:
                        merged_df = pd.merge(
                            merged_df, final_df,
                            how="outer",
                            left_index=True,
                            right_index=True
                        )

                # ===== FedMOpt =====
                temp_dfs = []
                for num_exec in range(1, num_execs + 1):
                    log_path = config['paths']['localLog'].format(
                        federation=federation,
                        strategy=final_strat,
                        sub_execution=f"{data_type}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                        num_exec=num_exec
                    )
                    time_path = config['paths']['localTimestamp'].format(
                        federation=federation,
                        strategy=final_strat,
                        sub_execution=f"{data_type}/{slow_dir_name}/beta{beta}/gamma{gamma}",
                        num_exec=num_exec
                    )

                    log_path = f"../{log_path}"
                    time_path = f"../{time_path}"

                    if os.path.exists(log_path):
                        temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                        temp_time_df = pd.read_csv(time_path)[["server_ev"]]
                    else:
                        temp_log_df = pd.DataFrame(columns=["server_ev_accuracy", "server_ev_loss"])
                        temp_time_df = pd.DataFrame(columns=["server_ev"])

                    temp_df = pd.concat([temp_log_df, temp_time_df], axis=1)
                    temp_df.columns = [
                        f'server_ev_accuracy_{num_exec}',
                        f'server_ev_loss_{num_exec}',
                        f'server_ev_time_{num_exec}'
                    ]
                    temp_dfs.append(temp_df)

                df_all = pd.concat(temp_dfs, axis=1)

                accuracy_cols = [col for col in df_all.columns if "accuracy" in col]
                loss_cols = [col for col in df_all.columns if "loss" in col]
                timestamp_cols = [col for col in df_all.columns if "time" in col]

                final_df = pd.DataFrame({
                    f'accuracy_{available + 1}': df_all[accuracy_cols].mean(axis=1),
                    f'loss_{available + 1}': df_all[loss_cols].mean(axis=1),
                    f'timestamp_{available + 1}': df_all[timestamp_cols].mean(axis=1),
                })

                merged_df = pd.merge(
                    merged_df, final_df,
                    how="outer",
                    left_index=True,
                    right_index=True
                )

                # ======================
                # SELECCIÓN DE MÉTODO
                # ======================
                if method == "FedMOpt":
                    loss_col = f"loss_{available + 1}"
                    time_col = f"timestamp_{available + 1}"
                else:
                    loss_col = f"loss_{method}"
                    time_col = f"timestamp_{method}"

                if loss_col not in merged_df:
                    continue

                plotted_any = True

                y_min_global = min(y_min_global, merged_df[loss_col].min())
                y_max_global = max(y_max_global, merged_df[loss_col].max())

                axs.plot(
                    merged_df[time_col],
                    merged_df[loss_col],
                    marker='o',
                    markersize=4,
                    linestyle='-',
                    label=rf"$\beta={beta},\ \gamma={gamma}$"
                )

            if not plotted_any:
                plt.close(fig)
                continue

            axs.set_ylim(y_min_global - 0.05, y_max_global + 0.05)
            axs.set_xlabel("Timestamp (s)", fontsize=18)
            axs.set_ylabel("Loss", fontsize=18)
            axs.legend(fontsize=12)
            axs.grid(True)
            axs.tick_params(axis='both', which='major', labelsize=16)

            title_method = f"sync{method}" if method != "FedMOpt" else "FedMOpt"

            plt.tight_layout()

            out_dir = f'../{federation}/results/{final_strat}/{data_type}/graphs/by_slow_method'
            os.makedirs(out_dir, exist_ok=True)

            plt.savefig(
                f'{out_dir}/{slow_dir_name}_{title_method}.pdf',
                format="pdf"
            )

            plt.close()"""

# ==============================
# BARPLOT TIEMPO - UNA GRÁFICA
# (beta fijo = 0.7)
# ==============================

methods = list(range(start, available + 1)) + ["FedMOpt"]  # sync5,6,7 + FedMOpt

# Diccionario para almacenar todos los tiempos
all_times = {}

for data_type in data_types:
    for slow in sorted(slows):
        if target_beta not in betas.get(slow, []):
            continue

        slow_dir_name = f"slow{slow[0]}"
        merged_df = None

        # ======================
        # CONSTRUCCIÓN merged_df (igual que antes)
        # ======================
        for sync_client in range(start, available + 1):
            temp_dfs = []

            for num_exec in range(1, num_execs + 1):
                log_path = config['paths']['localLog'].format(
                    federation=federation,
                    strategy=strategy,
                    sub_execution=f"{data_type}/sync{sync_client}/{slow_dir_name}/beta{target_beta}/gamma{target_gamma}",
                    num_exec=num_exec
                )
                time_path = config['paths']['localTimestamp'].format(
                    federation=federation,
                    strategy=strategy,
                    sub_execution=f"{data_type}/sync{sync_client}/{slow_dir_name}/beta{target_beta}/gamma{target_gamma}",
                    num_exec=num_exec
                )

                log_path = f"../{log_path}"
                time_path = f"../{time_path}"

                if os.path.exists(log_path):
                    temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                    temp_time_df = pd.read_csv(time_path)[["server_ev"]]
                else:
                    temp_log_df = pd.DataFrame(columns=["server_ev_accuracy", "server_ev_loss"])
                    temp_time_df = pd.DataFrame(columns=["server_ev"])

                temp_df = pd.concat([temp_log_df, temp_time_df], axis=1)
                temp_df.columns = [
                    f'server_ev_accuracy_{num_exec}',
                    f'server_ev_loss_{num_exec}',
                    f'server_ev_time_{num_exec}'
                ]
                temp_dfs.append(temp_df)

            df_all = pd.concat(temp_dfs, axis=1)

            final_df = pd.DataFrame({
                f'timestamp_{sync_client}': df_all.filter(like="time").mean(axis=1),
            })

            if merged_df is None:
                merged_df = final_df
            else:
                merged_df = pd.merge(
                    merged_df, final_df,
                    how="outer",
                    left_index=True,
                    right_index=True
                )

        # ===== FedMOpt =====
        temp_dfs = []
        for num_exec in range(1, num_execs + 1):
            log_path = config['paths']['localLog'].format(
                federation=federation,
                strategy=final_strat,
                sub_execution=f"{data_type}/{slow_dir_name}/beta{target_beta}/gamma{target_gamma}",
                num_exec=num_exec
            )
            time_path = config['paths']['localTimestamp'].format(
                federation=federation,
                strategy=final_strat,
                sub_execution=f"{data_type}/{slow_dir_name}/beta{target_beta}/gamma{target_gamma}",
                num_exec=num_exec
            )

            log_path = f"../{log_path}"
            time_path = f"../{time_path}"

            if os.path.exists(log_path):
                temp_time_df = pd.read_csv(time_path)[["server_ev"]]
            else:
                temp_time_df = pd.DataFrame(columns=["server_ev"])

            temp_time_df.columns = [f'server_ev_time_{num_exec}']
            temp_dfs.append(temp_time_df)

        df_all = pd.concat(temp_dfs, axis=1)

        final_df = pd.DataFrame({
            f'timestamp_{available + 1}': df_all.mean(axis=1),
        })

        merged_df = pd.merge(
            merged_df, final_df,
            how="outer",
            left_index=True,
            right_index=True
        )

        # ======================
        # EXTRAER TIEMPOS POR MÉTODO
        # ======================
        slow_times = {}
        for m in methods:
            if m == "FedMOpt":
                col = f"timestamp_{available + 1}"
                label = "FedMDyn"
            else:
                col = f"timestamp_{m}"
                label = f"sync{m}"

            if col in merged_df:
                last_val = merged_df[col].dropna()
                time = last_val.iloc[-1] if len(last_val) > 0 else 0.0
                slow_times[label] = time

        all_times[slow_dir_name] = slow_times

# ======================
# UNA SOLA GRÁFICA - BARRAS AGRUPADAS
# ======================
plt.figure(figsize=(10, 6))

# Preparar datos para barras agrupadas
slow_list = sorted(all_times.keys())
method_labels = [5, 6, 7, "FedMOpt"]
n_methods = len(method_labels)
bar_width = 0.2
x_pos = np.arange(len(slow_list))

# Crear barras agrupadas
for i, method in enumerate(method_labels):
    if i == n_methods - 1:
        times = [all_times[slow]["FedMDyn"] if "FedMDyn" in all_times[slow] else 0 
             for slow in slow_list]
        color = COLORS_BY_METHOD[method]
        label = "FedMDyn"
    else:
        times = [all_times[slow][f"sync{method}"] if f"sync{method}" in all_times[slow] else 0 
             for slow in slow_list]
        color = COLORS_BY_METHOD[f"sync{method}"]
        label = f"M = {method}"
        

    plt.bar(x_pos + i * bar_width, times, bar_width, 
            label=label, alpha=0.8, color=color)

# Personalizar gráfica
plt.xlabel("Configuración Slow", fontsize=18)
plt.ylabel("Tiempo total (s)", fontsize=18)
#plt.title(f"Tiempo total por método - {data_type.upper()} - beta={target_beta}", fontsize=18)
plt.xticks(x_pos + bar_width * (n_methods-1)/2, slow_list, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Guardar
out_dir = f'../{federation}/results/{final_strat}/{data_type}/graphs/barplots'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/time_comparison_all_slow_beta{target_beta}.pdf", format="pdf", bbox_inches='tight')
plt.close()

print(f"Gráfica guardada: time_comparison_all_slow_beta{target_beta}")