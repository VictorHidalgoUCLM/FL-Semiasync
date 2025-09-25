import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

strategies = ['FedAvg']
synces = [5]
data_types = ['iid']
window_sizes = [2048]

execution_times_path = '../local-execution/results/{strategy}/sync{sync}_data{data_type}_window{window_size}/total_times.csv'
timestamps_times_path = '../local-execution/results/{strategy}/sync{sync}_data{data_type}_window{window_size}/logs/timestamp_1.csv'

for strategy in strategies:
    for sync in synces:
        for data_type in data_types:
            combined_df = pd.DataFrame()
            total_times = []
            execution_times = []
            evaluation_times = []

            for window_size in window_sizes:
                execution_path = execution_times_path.format(strategy=strategy, sync=sync, data_type=data_type, window_size=window_size)
                timestamps_path = timestamps_times_path.format(strategy=strategy, sync=sync, data_type=data_type, window_size=window_size)

                df_exec = pd.read_csv(execution_path)
                df_exec['Window Size'] = window_size

                execution_times.append(df_exec['Total fit'].max())
                evaluation_times.append(df_exec['Total ev'].max())

                combined_df = pd.concat([combined_df, df_exec], ignore_index=True)

                df_timestamps = pd.read_csv(timestamps_path)
                total_times.append(df_timestamps['server_ev'].iloc[-1])

            pivoted_df = combined_df.pivot(index='Client', columns='Window Size', values='Total fit')

            plt.figure(figsize=(12, 8))
            x_positions = np.arange(len(pivoted_df.columns))
            bar_width = 0.15

            for i, node in enumerate(pivoted_df.index):
                plt.bar(x_positions + i * bar_width, pivoted_df.loc[node], width=bar_width, label=node)

            centered_positions = x_positions + bar_width * (len(pivoted_df.index) - 1) / 2

            # Plot total_times as a line
            plt.plot(centered_positions, np.array(total_times)-np.array(evaluation_times), marker='o', color='red', label='Communication times')
            plt.plot(centered_positions, execution_times, marker='o', color='purple', label='Execution times')

            plt.xlabel('Window Size')
            plt.ylabel('Time (seconds)')
            plt.title('Total, Execution and Evaluation Times vs Window Size')
            plt.xticks(x_positions + bar_width * (len(pivoted_df.index) - 1) / 2, pivoted_df.columns)
            plt.legend()
            plt.grid(True, axis='y')

            plt.savefig(f'../local-execution/results/{strategy}/sync{sync}_data{data_type}_time_analysis.png')
            plt.close() 


            comm_times = np.array(total_times)-np.array(evaluation_times)-np.array(execution_times)

            comm_percentage = np.multiply(100, np.divide(comm_times, total_times))
            fit_percentage = np.multiply(100, np.divide(execution_times, total_times))
            ev_percentage = np.multiply(100, np.divide(evaluation_times, total_times))

            data = {
                "Communication (%)": comm_percentage,
                "Fit (%)": fit_percentage,
                "Evaluation (%)": ev_percentage,
                "Total (%)": comm_percentage + fit_percentage + ev_percentage
            }

            df_percentages = pd.DataFrame(data, index=window_sizes)
            df_percentages.index.name = "Window size"

            print(df_percentages)
            print(total_times)