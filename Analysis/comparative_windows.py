import toml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_types = ['iid', 'noniid']
syncs_client = [5]
federations = ['local-execution']
strategies = ['FedAvg']
windows_sizes = [32, 64, 128, 256, 512, 1024, 1536, 2048]

devices = ['server', 'supernode-1', 'supernode-2', 'supernode-3', 'supernode-4', 'supernode-5']
scenarios = ['fit', 'ev']

inner_analysis = [f"{device}_{scenario}_accuracy" for device in devices for scenario in scenarios]

projectconf = '../projectconf.toml'

try:
    config = toml.load(projectconf)
except FileNotFoundError:
    print(f"El archivo {projectconf} no se encuentra.")
    exit(1)

for federation in federations:
    for strategy in strategies:
        for sync_client in syncs_client:
            for data_type in data_types:
                # Calculate the number of rows and columns for the subplots
                num_plots = len(windows_sizes)
                num_cols = 3  # You can adjust the number of columns as needed
                num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

                # Create a figure to hold all subplots
                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
                axes = axes.flatten()  # Flatten the axes array for easy iteration

                for ax, window_size in zip(axes, windows_sizes):
                    sub_execution = f"sync{sync_client}_data{data_type}_window{window_size}"

                    log_path = config['paths']['localLog'].format(federation=federation, strategy=strategy, sub_execution=sub_execution, num_exec=1)
                    log_path = f"../{log_path}"

                    temp_log_df = pd.read_csv(log_path)[inner_analysis]

                    # Plotting the data in the current subplot
                    for device in devices:
                        for scenario in scenarios:
                            column_name = f"{device}_{scenario}_accuracy"
                            if column_name in temp_log_df.columns:
                                ax.plot(temp_log_df[column_name], label=f"{device} ({scenario})")

                    ax.set_title(f"Window Size: {window_size}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel("Accuracy")
                    ax.legend()
                    ax.grid()

                # Hide any empty subplots
                for i in range(num_plots, len(axes)):
                    axes[i].axis('off')

                # Set the main title for the entire figure
                fig.suptitle(f"Accuracy Plots for {federation}, {strategy}, sync{sync_client}, data{data_type}")
                plt.tight_layout(rect=[0, 0, 1, 0.97])

                # Save the figure with all subplots
                plt.savefig(f"../{federation}/results/{strategy}/sync{sync_client}_data{data_type}.png", dpi=300)
                plt.close(fig)
