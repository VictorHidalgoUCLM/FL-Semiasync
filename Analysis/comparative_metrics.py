import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

# Define lists of the data that is going to be read and processed
# Loop order is -> strategy, sync, data_type, window_size
strategies = ['FedAvg_windows_sameSize']
synces = [3]
data_types = ['iid']
window_sizes = [32, 64, 128, 256, 512, 1024, 1536, 2048]
supernodes = range(1, 6)

# Unformatted path that reads metrics from a certain supernode at a certain configuration
metrics_unformatted_path = '../local-execution/results/{strategy}/sync{sync}_data{data_type}_window{window_size}/metrics/supernode-{supernode}/ex_1.csv'

# Dictionary that stores lists of rows to print later
dict_summary = {
    'rows_cpu': [],
    'rows_ram': [],
    'rows_net': [],
}

def plotFigure(data, x, y, hue, marker, title, xlabel, xticks, ylabel, legend, savefig, second_y=None):
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        marker=marker
    )

    if second_y is not None:
        sns.lineplot(
            data=data,
            x=x,
            y=second_y,
            hue=hue,
            marker=marker
        )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(ticks=xticks, rotation=50)
    plt.ylabel(ylabel)
    plt.legend(title=legend)
    plt.tight_layout()
    plt.savefig(savefig)
    plt.close()

# Iterate over different configurations, same as nested for loop
for strategy, sync, data_type in product(
    strategies, synces, data_types
):
    for supernode, window_size in product(
        supernodes, window_sizes
    ):
        # Format path having selected a configuration
        real_path = metrics_unformatted_path.format(
            strategy=strategy,
            sync=sync,
            data_type=data_type,
            window_size=window_size,
            supernode=supernode
        )

        # Save .csv to a dataframe and read necessary columns
        try:
            df = pd.read_csv(real_path)
        except FileNotFoundError:
            print(f"File not found: {real_path}")
            continue

        df = df[['Time_stamp(s)', 'CPU_time(s)', 'RAM_usage(B)', 'Net_transmit(B)', 'Net_receive(B)']]

        # Effective CPU usage = total CPU seconds usage / total app seconds
        last_time = df['Time_stamp(s)'].iloc[-1]    # Total app seconds
        last_cpu = df['CPU_time(s)'].iloc[-1]   # Total CPU seconds
        effective_cpu_usage = (last_cpu / last_time) * 100  # Effective CPU usage in %

        # Diff for CPU seconds usage (dCPU / dTime)
        d_cpu = df['CPU_time(s)'].diff()    # Differential CPU usage
        d_time = df['Time_stamp(s)'].diff() # Differential time elapsed between rows
        instantaneous_cpu = (d_cpu / d_time) * 100  # List of instantaneus CPU usage
        instantaneous_cpu = instantaneous_cpu.dropna()  # Drop NaN values

        # Save row for CPU usage
        row_cpu = {
            'Supernode': supernode,
            'Window Size': window_size,
            'Effective CPU (%)': round(effective_cpu_usage, 2),
            'Mean CPU (%)': round(instantaneous_cpu.mean(), 2),
            'Median CPU (%)': round(instantaneous_cpu.median(), 2),
            'Std Dev CPU (%)': round(instantaneous_cpu.std(), 2)
        }

        # Memory column cleaning (to MB)
        ram = df['RAM_usage(B)'] / (1024 * 1024)
        ram = ram.dropna()

        # Save row for RAM usage
        row_ram = {
            'Supernode': supernode,
            'Window Size': window_size,
            'Mean RAM (MB)': round(ram.mean(), 2),
            'Median RAM (MB)': round(ram.median(), 2),
            'Std Dev RAM (MB)': round(ram.std(), 2)
        }

        # Net transmit and receive columns cleaning (to MB)
        net_transmit = df['Net_transmit(B)'].iloc[-1] / (1024 * 1024)
        net_receive = df['Net_receive(B)'].iloc[-1] / (1024 * 1024)

        # Save row for Net usage
        row_net = {
            'Supernode': supernode,
            'Window Size': window_size,
            'Net Transmit (MB)': round(net_transmit, 2),
            'Net Receive (MB)': round(net_receive, 2)
        }

        # Save all data to dict
        dict_summary['rows_cpu'].append(row_cpu)
        dict_summary['rows_ram'].append(row_ram)
        dict_summary['rows_net'].append(row_net)

    summary_cpu = pd.DataFrame(dict_summary['rows_cpu'])
    summary_ram = pd.DataFrame(dict_summary['rows_ram'])
    summary_net = pd.DataFrame(dict_summary['rows_net'])

    print(summary_cpu)
    print(summary_net)
    print(summary_ram)

    sns.set_theme(style='whitegrid')

    plotFigure(summary_cpu, 'Window Size', 'Mean CPU (%)', 'Supernode', 'o',
               "Mean CPU usage vs Window size per supernode", "Window Size", window_sizes,
               "Mean CPU (%)", "Supernode", f'../local-execution/results/{strategy}/sync{sync}_data{data_type}_metrics_CPU.png')

    plotFigure(summary_ram, 'Window Size', 'Mean RAM (MB)', 'Supernode', 'o',
               "Mean RAM usage vs Window size per supernode", "Window Size", window_sizes,
               "Mean RAM (MB)", "Supernode", f'../local-execution/results/{strategy}/sync{sync}_data{data_type}_metrics_RAM.png')
    
    plotFigure(summary_net, 'Window Size', 'Net Transmit (MB)', 'Supernode', 'o',
               "Net usage vs Window size per supernode", "Window Size", window_sizes,
               "Net Usage (MB)", "Supernode", f'../local-execution/results/{strategy}/sync{sync}_data{data_type}_metrics_Net.png',
               "Net Receive (MB)")