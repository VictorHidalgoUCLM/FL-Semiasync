import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import toml

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
data_types = ['noniid']
strategies = ['FedAvg']
federations = ['local-execution']
slows = [2]

num_execs = 1


acc_columns = ['loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5', 'loss_6', 'loss_7', 'loss_8', 'loss_9']
time_columns = ['timestamp_1', 'timestamp_2', 'timestamp_3', 'timestamp_4', 'timestamp_5', 'timestamp_6', 'timestamp_7', 'timestamp_8', 'timestamp_9']

og_colors = plt.cm.tab10.colors

y_min_global = np.inf
y_max_global = -np.inf

try:
    config = toml.load(projectconf)
except FileNotFoundError:
    print(f"El archivo {projectconf} no se encuentra. Asegúrate de que exista.")
    exit(1)

for federation in federations:
    results_list = []
    time_objective = []
    for strategy in strategies:
        for data_type in data_types:
            for heterogeneity in ['homogeneous']:
                for slow in slows:
                    merged_df = None

                    for sync_client in range(1, 9):
                        log_path = config['paths']['localLog'].format(federation=federation, strategy=strategy, sub_execution=f"sync{sync_client}_data{data_type}_{heterogeneity}", num_exec=1)
                        time_path = config['paths']['localTimestamp'].format(federation=federation, strategy=strategy, sub_execution=f"sync{sync_client}_data{data_type}_{heterogeneity}", num_exec=1)

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
                        log_path = config['paths']['localLog'].format(federation=federation, strategy='FedMOpt', sub_execution=f"slow{slow}_data{data_type}_{heterogeneity}", num_exec=num_exec+1)
                        time_path = config['paths']['localTimestamp'].format(federation=federation, strategy='FedMOpt', sub_execution=f"slow{slow}_data{data_type}_{heterogeneity}", num_exec=num_exec+1)

                        log_path = f"../{log_path}"
                        time_path = f"../{time_path}"

                        metric = pd.read_csv(log_path)[["server_ev_loss"]].to_numpy().ravel()
                        time = pd.read_csv(time_path)[["server_ev"]].to_numpy().ravel()

                        exec_metrics.append(metric)
                        exec_times.append(time)
                    
                    print(exec_metrics)

                    t_max = min(time[-1] for time in exec_times)
                    t_grid = np.linspace(0, t_max, 50)

                    all_interp = []
                    for time, metric in zip(exec_times, exec_metrics):
                        interp = np.interp(t_grid, time, metric)
                        all_interp.append(interp)

                    acc_mean = np.mean(all_interp, axis=0)

                    final_df = pd.DataFrame({
                        f'loss_9': acc_mean,
                        f'timestamp_9': t_grid
                    })
                    merged_df = pd.merge(merged_df, final_df, how="outer", left_index=True, right_index=True)

                    for i in range(1, 10):
                        # Calculamos el mínimo y máximo de loss para establecer el rango global
                        y_min_global = min(y_min_global, merged_df[f"loss_{i}"].min())
                        y_max_global = max(y_max_global, merged_df[f"loss_{i}"].max())

                    last_valids = merged_df[time_columns].apply(lambda col: col.dropna().iloc[-1])
                    max_time = last_valids.min()
                    time_objective.append(max_time)

                    estimations = []

                    for timestamp, loss in zip(time_columns, acc_columns):
                        temp_df = merged_df[[timestamp]]

                        if max_time in temp_df[timestamp].values:
                            index_below = len(temp_df) - 1
                            index_above = index_below

                        else:
                            index_below = temp_df[temp_df[timestamp] <= max_time].index[-1]
                            index_above = temp_df[temp_df[timestamp] > max_time].index[0]

                        y_below = merged_df[loss].loc[index_below]
                        y_above = merged_df[loss].loc[index_above]

                        x_below = merged_df[timestamp].loc[index_below]
                        x_above = merged_df[timestamp].loc[index_above]

                        if index_below == index_above:
                            estimation = y_below
                        else:
                            estimation = y_below + ((max_time - x_below) * (y_above - y_below) / (x_above - x_below))

                        estimations.append(estimation)
                    
                    simplified_df = pd.Series(estimations, index=acc_columns)
                    simplified_df[acc_columns] = simplified_df[acc_columns].div(simplified_df['loss_8'], axis=0)
                    simplified_df[acc_columns] = simplified_df[acc_columns] - 1
                    simplified_df[acc_columns] = simplified_df[acc_columns] * 100
                    simplified_df[acc_columns] = simplified_df[acc_columns].round(2)
                    simplified_df = simplified_df.drop('loss_9')

                    results_list.append(simplified_df)

                    fig, axs = plt.subplots(1, 1, figsize=(10, 6))  # 'sharex=True' para compartir el eje x (timestamp)

                    for i in range(1, 10):
                        if i == 9:
                            axs.plot(merged_df[f"timestamp_{i}"], merged_df[f"loss_{i}"], marker='o', markersize=4, linestyle='-', label=f"FedMOpt", color=og_colors[i-1])
                        else:
                            axs.plot(merged_df[f"timestamp_{i}"], merged_df[f"loss_{i}"], marker='o', markersize=4, linestyle='-', label=f"M = {i}", color=og_colors[i-1])

                    # Establece los límites del eje Y usando el rango global calculado
                    axs.set_ylim(y_min_global - 0.1, y_max_global + 0.1)  # Un margen pequeño para mayor claridad

                    axs.set_xlabel("Timestamp (s)", fontsize=18)
                    axs.set_ylabel("Loss", fontsize=18)
                    axs.axvline(x=max_time, color='black', linestyle='--')
                    axs.legend(fontsize=16)
                    axs.grid(True)

                    axs.tick_params(axis='both', which='major', labelsize=16)

                    plt.tight_layout()

                    plt.savefig(f'../{federation}/results/{strategy}/results_slow{slow}_data{data_type}_{heterogeneity}.png', dpi=600)

                    
                    loss_cols = [col for col in merged_df.columns if "loss" in col]
                    
                    print(f"Execution {slow}")
                    for loss_col in loss_cols:
                        # índice de la fila donde ocurre el mínimo de la columna loss
                        idx_min = merged_df[loss_col].idxmin()
                        
                        # nombre de la columna timestamp correspondiente
                        n = loss_col.split("_")[1]         # extrae el número
                        timestamp_col = f"timestamp_{n}"
                        
                        # valor del timestamp en esa fila
                        ts_value = merged_df.at[idx_min, timestamp_col]
                        min_loss = merged_df.at[idx_min, loss_col]

                        score = 1 / (min_loss * ts_value) * 1e4

                        print(f"{loss_col}: min = {merged_df.at[idx_min, loss_col]} en timestamp = {ts_value}. Score: {score}")
