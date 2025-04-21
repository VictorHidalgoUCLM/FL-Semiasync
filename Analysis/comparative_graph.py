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
data_types = ['iid', 'noniid']
strategies = ['FedAvg_homogeneous_good', 'FedAvg_heterogeneous_good']
federations = ['local-execution']

acc_columns = ['accuracy_1', 'accuracy_2', 'accuracy_3', 'accuracy_4', 'accuracy_5']
time_columns = ['timestamp_1', 'timestamp_2', 'timestamp_3', 'timestamp_4', 'timestamp_5']

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
            merged_df = None

            for sync_client in [1, 2, 3, 4, 5]:
                log_path = config['paths']['localLog'].format(federation=federation, strategy=strategy, sub_execution=f"sync{sync_client}_data{data_type}", num_exec=1)
                time_path = config['paths']['localTimestamp'].format(federation=federation, strategy=strategy, sub_execution=f"sync{sync_client}_data{data_type}", num_exec=1)

                temp_log_df = pd.read_csv(log_path)[["server_ev_accuracy", "server_ev_loss"]]
                temp_time_df = pd.read_csv(time_path)[["server_ev"]]

                final_df = pd.concat([temp_log_df, temp_time_df], axis=1)
                final_df.columns = [f'accuracy_{sync_client}', f'loss_{sync_client}', f'timestamp_{sync_client}']

                if merged_df is None:
                    merged_df = final_df
                else:
                    merged_df = pd.merge(merged_df, final_df, how="outer", left_index=True, right_index=True)

            for i in range(1, 6):
                # Calculamos el mínimo y máximo de accuracy para establecer el rango global
                y_min_global = min(y_min_global, merged_df[f"accuracy_{i}"].min())
                y_max_global = max(y_max_global, merged_df[f"accuracy_{i}"].max())

            max_time = merged_df[time_columns].iloc[-1].min()
            time_objective.append(max_time)

            estimations = []

            for timestamp, accuracy in zip(time_columns, acc_columns):
                temp_df = merged_df[[timestamp]]

                if max_time in temp_df[timestamp].values:
                    index_below = len(temp_df) - 1
                    index_above = index_below

                else:
                    index_below = temp_df[temp_df[timestamp] <= max_time].index[-1]
                    index_above = temp_df[temp_df[timestamp] > max_time].index[0]

                y_below = merged_df[accuracy].loc[index_below]
                y_above = merged_df[accuracy].loc[index_above]

                x_below = merged_df[timestamp].loc[index_below]
                x_above = merged_df[timestamp].loc[index_above]

                if index_below == index_above:
                    estimation = y_below
                else:
                    estimation = y_below + ((max_time - x_below) * (y_above - y_below) / (x_above - x_below))

                estimations.append(estimation)
            
            simplified_df = pd.Series(estimations, index=acc_columns)
            simplified_df[acc_columns] = simplified_df[acc_columns].div(simplified_df['accuracy_5'], axis=0)
            simplified_df[acc_columns] = simplified_df[acc_columns] - 1
            simplified_df[acc_columns] = simplified_df[acc_columns] * 100
            simplified_df[acc_columns] = simplified_df[acc_columns].round(2)
            simplified_df = simplified_df.drop('accuracy_5')

            results_list.append(simplified_df)

            fig, axs = plt.subplots(1, 1, figsize=(10, 6))  # 'sharex=True' para compartir el eje x (timestamp)

            for i in range(1, 6):
                axs.plot(merged_df[f"timestamp_{i}"], merged_df[f"accuracy_{i}"], marker='o', markersize=4, linestyle='-', label=f"M = {i}", color=og_colors[i-1])

            # Establece los límites del eje Y usando el rango global calculado
            axs.set_ylim(y_min_global - 0.1, y_max_global + 0.1)  # Un margen pequeño para mayor claridad

            axs.set_xlabel("Timestamp (s)")
            axs.set_ylabel("Accuracy")
            axs.axvline(x=max_time, color='black', linestyle='--')
            axs.legend()
            axs.grid(True)

            plt.tight_layout()

            plt.savefig(f'../{federation}/results/{strategy}/results_{data_type}.png')

    final_results_df = pd.DataFrame(results_list)
    final_results_df.index = [f'{strategy}_{data_type}' for strategy in strategies for data_type in data_types]
    final_results_df.columns = [f'M={i} vs M=5' for i in [1, 2, 3, 4]]

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis('off')

    table = ax.table(cellText=final_results_df.values, colLabels=final_results_df.columns, rowLabels=final_results_df.index, loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    table.auto_set_column_width(col=list(range(len(final_results_df.columns))))

    for i, col in enumerate(table.get_celld().values()):
        if i < len(final_results_df.columns):  # Asegurarse de que estamos trabajando con las celdas de la cabecera de la columna
            cell = table[(0, i)]  # Seleccionar la celda en la primera fila (encabezados de columnas)
            cell.set_facecolor(og_colors[i])  # Asignar el color de fondo
            cell.set_text_props(ha='center', va='center')  # Centrar el texto en las celdas de la cabecera de las columnas


    for i, (row, col) in enumerate([(i, j) for i in range(len(final_results_df.index)) for j in range(len(final_results_df.columns))]):
        cell = table[(row + 1, col)]  # Ajustar para las celdas de datos (empezando desde la fila 1)
        if (row + 1) % 2 == 0:
            colors = [lighten_color(color, factor=0.5) for color in og_colors]
        else:
            colors = [lighten_color(color, factor=0.3) for color in og_colors]

        original_text = final_results_df.iloc[row, col]

        text = cell.get_text()
        text.set_text(f"{original_text} %")

        cell.set_facecolor(colors[i % (len(acc_columns)-1)])  # Colores cíclicos
        # Centrar el texto en las celdas de datos
        cell.set_text_props(ha='center', va='center')

    for i in range(len(final_results_df.columns)):
        if (i+1) % 2 == 0:
            cell = table[(i+1, -1)]
            cell.set_facecolor(darken_color((1, 1, 1), factor=0.75))

    plt.savefig(f'{federation}/results', dpi=1000)