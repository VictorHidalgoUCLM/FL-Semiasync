import toml
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import numpy as np

data_types = ['iid']
syncs_client = [5]
federations = ['local-execution']
strategies = ['FedAvg']
windows_sizes = [32, 64, 128, 256, 512, 1024, 1536, 2048]

devices = ['server']#, 'supernode-1', 'supernode-2', 'supernode-3', 'supernode-4', 'supernode-5']
scenarios = ['fit', 'ev']

inner_analysis = [
    f"{device}_{scenario}_{metric}"
    for device in devices
    for scenario in scenarios
    for metric in ['accuracy', 'loss']
]

projectconf = '../projectconf.toml'

try:
    config = toml.load(projectconf)
except FileNotFoundError:
    print(f"El archivo {projectconf} no se encuentra.")
    exit(1)

def get_global_loss_range(config, windows_sizes, federation, strategy, sync_client, data_type):
    min_loss, max_loss = float('inf'), float('-inf')

    for window_size in windows_sizes:
        sub_execution = f"sync{sync_client}_data{data_type}_window{window_size}"
        log_path = config['paths']['localLog'].format(
            federation=federation, strategy=strategy, sub_execution=sub_execution, num_exec=1
        )
        log_path = f"../{log_path}"

        try:
            df = pd.read_csv(log_path)
            for device in devices:
                col = f"{device}_ev_loss"
                if col in df.columns:
                    min_loss = min(min_loss, df[col].min())
                    max_loss = max(max_loss, df[col].max())
        except:
            continue

    return min_loss, max_loss

global_min_loss, global_max_loss = float('inf'), float('-inf')

for federation, strategy, sync_client, data_type, window_size in product(
     federations, strategies, syncs_client, data_types, windows_sizes
):
    sub_execution = f"sync{sync_client}_data{data_type}_window{window_size}"
    log_path = config['paths']['localLog'].format(
        federation=federation, strategy=strategy, sub_execution=sub_execution, num_exec=1
    )
    log_path = f"../{log_path}"
    try:
        df = pd.read_csv(log_path)
        for device in devices:
            loss_col = f"{device}_ev_loss"
            if loss_col in df.columns:
                global_min_loss = min(global_min_loss, df[loss_col].min())
                global_max_loss = max(global_max_loss, df[loss_col].max())
    except Exception as e:
        print(f"Error leyendo {log_path}: {e}")

for federation, strategy, sync_client, data_type in product(
     federations, strategies, syncs_client, data_types
):
    # Calculate the number of rows and columns for the subplots
    num_plots = len(windows_sizes)
    num_cols = 3  # You can adjust the number of columns as needed
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

    # Create a figure to hold all subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows), sharey=True)
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, window_size in zip(axes, windows_sizes):
        sub_execution = f"sync{sync_client}_data{data_type}_window{window_size}"
        log_path = config['paths']['localLog'].format(
            federation=federation, strategy=strategy, sub_execution=sub_execution, num_exec=1
        )
        log_path = f"../{log_path}"

        temp_log_df = pd.read_csv(log_path)[inner_analysis]

        ax2 = ax.twinx()  # Segundo eje y para 'loss'
        ax2.set_ylim(global_min_loss, global_max_loss)

        for device in devices:
            for scenario in scenarios:
                accuracy_column = f"{device}_{scenario}_accuracy"
                loss_column = f"{device}_{scenario}_loss"

                if accuracy_column in temp_log_df.columns:
                    ax.plot(temp_log_df[accuracy_column], label=f"{device} ({scenario}) - acc")

                if scenario != 'fit' and loss_column in temp_log_df.columns:
                    ax2.plot(temp_log_df[loss_column], label=f"{device} ({scenario}) - loss", color='green')

        ax.set_title(f"Window Size: {window_size}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Accuracy")
        ax2.set_ylabel("Loss")
        ax.grid()

        # Combinar leyendas de ambos ejes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='best')


    # Hide any empty subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    # Set the main title for the entire figure
    fig.suptitle(f"Accuracy Plots for {federation}, {strategy}, sync{sync_client}, data{data_type}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the figure with all subplots
    plt.savefig(f"../{federation}/results/{strategy}/sync{sync_client}_data{data_type}_byround.png", dpi=300)
    plt.close(fig)


fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

colors = plt.cm.viridis(np.linspace(0, 1, len(windows_sizes)))


for i, (federation, strategy, sync_client, data_type) in enumerate(product(federations, strategies, syncs_client, data_types)):
    for j, window_size in enumerate(windows_sizes):
        sub_execution = f"sync{sync_client}_data{data_type}_window{window_size}"

        # Load paths
        log_path = f"../{config['paths']['localLog'].format(federation=federation, strategy=strategy, sub_execution=sub_execution, num_exec=1)}"
        timestamp_path = f"../{config['paths']['localTimestamp'].format(federation=federation, strategy=strategy, sub_execution=sub_execution, num_exec=1)}"

        try:
            temp_loss = pd.read_csv(log_path)['server_ev_loss']
            temp_time = pd.read_csv(timestamp_path)['server_ev']
        except FileNotFoundError:
            print(f"Saltando {sub_execution}, archivo no encontrado.")
            continue

        # Plot accuracy
        ax1.plot(temp_time, label=f'Acc Window {window_size}', color=colors[j], linestyle='-')

        # Plot loss
        ax2.plot(temp_loss, label=f'Loss Window {window_size}', color=colors[j], linestyle='--')

# Etiquetas y leyenda
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy", color='blue')
ax2.set_ylabel("Loss", color='green')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', bbox_to_anchor=(1.0, 0.5))

plt.title("Accuracy & Loss vs Window Size")
plt.tight_layout()
plt.grid(True)

plt.savefig("all_window_sizes_plot.png", dpi=300)
plt.close(fig)
