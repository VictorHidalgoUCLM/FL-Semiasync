import pandas as pd
import matplotlib.pyplot as plt

# Parámetros
strategies = ['FedAvg']
synces = [5]
data_types = ['iid']
window_sizes = [32, 64, 128, 256, 512, 1024, 1536, 2048]
supernodes = range(1, 6)
metrics_unformatted_path = '../local-execution/results/{strategy}/sync{sync}_data{data_type}_window{window_size}/metrics/supernode-{supernode}/ex_1.csv'

# Diccionario para guardar CPU usage por supernodo
summary_rows_cpu = []
summary_rows_ram = []
summary_rows_net = []

# Iterar sobre combinaciones
for strategy in strategies:
    for sync in synces:
        for data_type in data_types:
            for window_size in window_sizes:
                supernode = 1
                real_path = metrics_unformatted_path.format(
                    strategy=strategy,
                    sync=sync,
                    data_type=data_type,
                    window_size=window_size,
                    supernode=supernode
                )
                df = pd.read_csv(real_path)
                df = df[['Time_stamp(s)', 'CPU_time(s)', 'RAM_usage(B)', 'Net_transmit(B)', 'Net_receive(B)']]

                # Effective CPU usage
                last_time = df['Time_stamp(s)'].iloc[-1]
                last_cpu = df['CPU_time(s)'].iloc[-1]
                effective_cpu_usage = (last_cpu / last_time) * 100

                # Derivar uso instantáneo (dCPU / dTime)
                d_cpu = df['CPU_time(s)'].diff()
                d_time = df['Time_stamp(s)'].diff()
                instantaneous_cpu = (d_cpu / d_time) * 100

                # Evitar NaN del primer valor
                instantaneous_cpu = instantaneous_cpu.dropna()

                row_cpu = {
                    'Window Size': window_size,
                    'Effective CPU (%)': round(effective_cpu_usage, 2),
                    'Mean CPU (%)': round(instantaneous_cpu.mean(), 2),
                    'Median CPU (%)': round(instantaneous_cpu.median(), 2),
                    'Std Dev CPU (%)': round(instantaneous_cpu.std(), 2)
                }

                summary_rows_cpu.append(row_cpu)

                
                ram = df['RAM_usage(B)'] / (1024 * 1024)  # Convertir a MB
                ram = ram.dropna()

                row_ram = {
                    'Window Size': window_size,
                    'Mean RAM (MB)': round(ram.mean(), 2),
                    'Median RAM (MB)': round(ram.median(), 2),
                    'Std Dev RAM (MB)': round(ram.std(), 2)
                }

                summary_rows_ram.append(row_ram)


                net_transmit = df['Net_transmit(B)'].iloc[-1] / (1024 * 1024)
                net_receive = df['Net_receive(B)'].iloc[-1] / (1024 * 1024)

                row_net = {
                    'Window Size': window_size,
                    'Net Transmit (MB)': round(net_transmit, 2),
                    'Net Receive (MB)': round(net_receive, 2)
                }

                summary_rows_net.append(row_net)

summary_cpu = pd.DataFrame(summary_rows_cpu)
summary_ram = pd.DataFrame(summary_rows_ram)
summary_net = pd.DataFrame(summary_rows_net)

print(summary_cpu.to_string(index=False))
print(summary_ram.to_string(index=False))
print(summary_net.to_string(index=False))
