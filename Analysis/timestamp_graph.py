import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import toml
import os
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--federation", 
    type=str, 
    choices=["local-execution", "remote-execution"], 
    help="Tipo de ejecución: local-execution | remote-execution",
    required=True
)

parser.add_argument(
    "--strategy", 
    type=str, 
    required=True
)

parser.add_argument(
    '-s',
    '--sync_client',
    type=int,
    required=True
)

parser.add_argument(
    '-t',
    '--data_type',
    type=str,
    required=True,
)

parser.add_argument(
    '-n',
    '--number_execution',
    type=int,
    help="Cantidad de ejecuciones a realizar",
    required=True
)

parser.add_argument(
    '-w',
    '--window_size',
    type=int,
    help="Cantidad de ejecuciones a realizar",
    required=True
)

def main():
    args = parser.parse_args()

    color_inicio = np.array([255, 204, 0]) / 255
    color_fin = np.array([153, 51, 0]) / 255

    projectconf = 'projectconf.toml'

    federation = args.federation
    sync_client = args.sync_client
    strategy = args.strategy
    data_type = args.data_type
    number_execution = args.number_execution
    window_size = args.window_size

    execution_name = f"sync{sync_client}_data{data_type}_window{window_size}"

    try:
        config = toml.load(projectconf)
    except FileNotFoundError:
        print(f"El archivo {projectconf} no se encuentra. Asegúrate de que exista.")
        exit(1)

    file_path = config['paths']['localTimestamp'].format(
            federation=federation,
            strategy=strategy,
            sub_execution=execution_name,
            num_exec=number_execution
        )

    df = pd.DataFrame(columns=['round', 'device', 'epoch', 'timestamp', 'time'])
    df['time'] = pd.to_numeric(df['time'], errors='coerce')


    with open(file_path, 'r') as file:
        for row_lane, row in enumerate(file.readlines()):
            lane = row.strip().split(",")

            if row_lane == 0:
                headers = lane

            else:
                for column_lane, column in enumerate(lane):
                    round, epoch = headers[column_lane].split("_")
                    
                    for k, value in enumerate(column.split(";")):
                        df = df._append({
                            'round': row_lane,
                            'device': round,
                            'epoch': epoch,
                            'timestamp': k,
                            'time': float(value)
                        }, ignore_index=True)

    server_df = df[(df['device'] == 'server')].drop(columns=['device', 'timestamp'])
    server_fit = server_df[server_df['epoch']=='fit']['time']
    server_ev = server_df[server_df['epoch']=='ev']['time']

    n_clientes = len(config['devices'])  # Número de clientes
    n_waiting = sync_client
    tiempo_maximo = 300  # Todos los clientes terminan en el mismo tiempo

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Graficar cada cliente, todos empiezan en 0 y terminan en el mismo tiempo
    for i in range(n_clientes):
        ax.plot([0, tiempo_maximo], [i, i], color='black', label=f'Cliente {i+1}' if i == 0 else "")

    ymin, ymax = ax.get_ylim()

    # Draw squares (or rectangles) for each supernode
    square_size = 0.40  # Define the size of the square, e.g., 10 units
    start_time_round = 0

    sum = {}

    for round, group_round in df.groupby('round'):
        supernodes_fit = {}
        supernodes_ev = {}

        end_fit, end_ev = group_round[group_round['device'] == 'server']['time']
        group_round = group_round[group_round['device'] != 'server']

        times_to_sort = sorted(group_round[group_round['epoch'] == 'fit']['time'])[:-n_clientes]

        mid_fits = times_to_sort[n_waiting-1::n_waiting] 
        gradiente = np.zeros((len(mid_fits)+1, 3), dtype=float)
        hatches = [
            '////', '\\\\\\\\', '||||', '----', '++++', 'xxxx', 'oooo', '....',
            '//--', '\\\\++', 'xx||', 'o-|x', '/o-x', '/\\|-', '//xx', '--++'
        ]

        for l in range(len(mid_fits)+1):
            dividendo = len(mid_fits) - 1 if len(mid_fits) > 1 else 1
            gradiente[l] = np.clip(color_inicio + (color_fin - color_inicio) * (l / dividendo), 0, 1)

        for i, (timestamp, group_timestamp) in enumerate(group_round.groupby('timestamp')):   
            for device, group_device in group_timestamp.groupby('device'):
                if i == 0:
                    supernodes_fit[device] = [(start_time_round, group_timestamp[(group_timestamp['epoch'] == 'fit') & (group_timestamp['device'] == device)]['time'].iloc[0], color_inicio, hatches[0])]  
                    supernodes_ev[device] = [(end_fit, group_timestamp[(group_timestamp['epoch'] == 'ev') & (group_timestamp['device'] == device)]['time'].iloc[0])]         
                else:
                    _, prev_res, _, _ = supernodes_fit[device][i-1]
                    aux_fit = start_time_round
                        
                    for j, (mid_fit) in enumerate(mid_fits):
                        if aux_fit < prev_res <= mid_fit:
                            supernodes_fit[device].append((mid_fit, group_timestamp[(group_timestamp['epoch'] == 'fit') & (group_timestamp['device'] == device)]['time'].iloc[0], gradiente[j+1], hatches[j+1]))           
                            aux_fit = mid_fit

        start_time_round = end_ev

        for client, times in supernodes_fit.items():
            sum.setdefault(client, 0)

            for start_time, end_time, color, hatch in times:
                sum[client] += end_time - start_time

        for i, (_, times) in enumerate(supernodes_fit.items()):
            y_pos = i
            y_pos_centered = y_pos-square_size/2

            for start_time, end_time, color, hatch in times:
                # Add a rectangle for the square (this is a series of squares, you can adjust as needed)
                ax.add_patch(patches.Rectangle(
                        (start_time, y_pos_centered),  # (x, y) position
                        end_time - start_time,  # width of the square
                        square_size,  # height (fixed in this case)
                        edgecolor='black',  # Edge color
                        facecolor=color,  # Fill color
                        hatch=hatch,
                        alpha=0.5
                    )
                )

        for i, (client, times) in enumerate(supernodes_ev.items()):
            y_pos = i
            y_pos_centered = y_pos-square_size/2

            for start_time, end_time in times:
                # Add a rectangle for the square (this is a series of squares, you can adjust as needed)
                ax.add_patch(patches.Rectangle(
                        (start_time, y_pos_centered),  # (x, y) position
                        end_time - start_time,  # width of the square
                        square_size,  # height (fixed in this case)
                        edgecolor='black',  # Edge color
                        facecolor='turquoise',  # Fill color
                        alpha=0.5  # Transparency
                    )
                )

        for timestamp in mid_fits:
            ax.vlines(timestamp, ymin=ymin, ymax=ymax, color='black', linestyle='dotted', alpha=0.5)

    totaltime_path = f'{os.path.dirname(os.path.dirname(file_path))}/total_times.csv'

    with open(totaltime_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Client', 'Total time'])

        for client, total_time in sum.items():
            writer.writerow([client, total_time])

    # Configuración de los ejes
    ax.set_xlabel('Execution time', fontsize=16)
    ax.set_yticks(range(n_clientes))
    ax.set_yticklabels([f'Client-{i+1}' for i in range(n_clientes)], fontsize=16)
    #ax.set_title(f'Ejecución FL', fontsize=20)
    ax.set_xlim(0, tiempo_maximo)
    ax.vlines(server_fit, ymin=ymin, ymax=ymax, color='red', linestyle=(0, (5,2,1,2)), label='fit', linewidth=2)
    ax.vlines(server_ev, ymin=ymin, ymax=ymax, color='green', linestyle=(0, (5,2)), label='ev', linewidth=2)
    ax.tick_params(axis='x', labelsize=14)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.savefig(f'{os.path.dirname(os.path.dirname(file_path))}/timestamps.png', dpi=1000)

if __name__ == '__main__':
    main()