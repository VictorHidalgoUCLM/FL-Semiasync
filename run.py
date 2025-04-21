import subprocess
import argparse
import threading
import toml
import signal
import time

import os
import fnmatch
import re
import requests

projectconf = 'projectconf.toml'
handler_flag = False
INTERVAL = 10

# Configuration to delete data series on the server
url = 'http://localhost:9090/api/v1/admin/tsdb/delete_series'
params = {'match[]': '{job="cadvisor"}'}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-f",
    "--federation", 
    type=str, 
    nargs='+',
    choices=["local-execution", "remote-execution"], 
    help="Tipo de ejecución: local-execution | remote-execution",
    default=["local-execution"],
    required=False
)

parser.add_argument(
    '-d',
    '--debug',
    action='store_true',
    help="Ejecución rápida de Flwr, hacemos forward de parámetros sin entrenar",
    required=False
)

parser.add_argument(
    '-s',
    '--sync_clients',
    type=int,
    nargs='+',
    help="Lista del número de clientes con los que se sincronizan internamente las rondas de entrenamiento",
    default=[3],
    required=False
)

parser.add_argument(
    '-t',
    '--data_list',
    type=str,
    nargs='+',
    help="Lista de tipos de datos a ejecutar",
    required=False,
    default=['iid']
)

parser.add_argument(
    '-n',
    '--number_execution',
    type=int,
    help="Cantidad de ejecuciones a realizar",
    required=False,
    default=1
)

def signal_handler(sig, frame, event):
    """
    Receives the signal to terminate the program, activates the event to 
    terminate all active threads.
    """
    global handler_flag

    event.set()
    handler_flag = True


def check_docker_logs(event, federation, strategy, execution_name, num_exec):
    pattern_found_event = threading.Event()
    PATTERN = f"Training {federation} {strategy} {execution_name} {num_exec} ended"

    while not pattern_found_event.is_set() and not event.is_set():
        try:
            # Ejecuta el comando `docker logs` y obtiene la última línea
            result = subprocess.run(
                ["docker", "logs", "--tail", "4", "serverapp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Verifica si la última línea contiene el patrón deseado
            if PATTERN in result.stdout.strip():
                pattern_found_event.set()

        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar el comando: {e}")

        # Espera el intervalo especificado
        time.sleep(INTERVAL)


def local_run(command, show_output=False):
    try:
        command_parts = command.split()
        print(f"\nEjecutando comando {command}...\n")

        if show_output:
            subprocess.run(command_parts, check=True)
        else:
            subprocess.run(command_parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando Docker: {e}")
    

def local_Popen(command, show_output=False):
    try:
        command_parts = command.split()
        print(f"\nEjecutando comando {command}...\n")

        if show_output:
            process = subprocess.Popen(command_parts)
        else:
            process = subprocess.Popen(command_parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando Docker: {e}")

    return process


def remote_run(command, ip, user):
    try:
        ssh_command = ["ssh", f"{user}@{ip}", command]
        print(f"Ejecutando comando remoto: {ssh_command}")
        subprocess.run(ssh_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    except subprocess.CalledProcessError as e:
        pass


def stop_remote_code(id, user, ip):
    stop_supernode = ["docker", "stop", f"supernode-{id}"]
    stop_clientapp = ["docker", "stop", f"client-{id}"]
    stop_network = ["docker", "network", "rm", "flwr-network"]

    remote_supernode = f'ssh {user}@{ip} "{" ".join(stop_supernode)}"'
    remote_clientapp = f'ssh {user}@{ip} "{" ".join(stop_clientapp)}"'
    remote_network = f'ssh {user}@{ip} "{" ".join(stop_network)}"'

    subprocess.run(remote_supernode, shell=True, capture_output=True, text=True)
    subprocess.run(remote_clientapp, shell=True, capture_output=True, text=True)
    subprocess.run(remote_network, shell=True, capture_output=True, text=True)


def update_clients():
    local_run("docker buildx build -f dockerfiles/clientapp.Dockerfile --platform linux/arm64/v8 -t flwr_clientapp:latest .", True)
    local_run("docker tag flwr_clientapp:latest victorhidalgo/clientapp")
    local_run("docker push victorhidalgo/clientapp")


def get_last_round(config, strategy_name, num_exec, federation, name, wait=False):
    """
    Auxiliary function to know how many rounds are left to execute in case 
    the program terminated abruptly, returning the next round to execute.
    """
    directory_name = config['paths']['localCheckpoint'].format(strategy=strategy_name, num_exec=num_exec, federation=federation, sub_execution=name)

    os.makedirs(directory_name, exist_ok=True)

    file_pattern = "round-*-weights.npz"
    timeout = 50
    start_time = time.time()

    while True:
        files = [file for file in os.listdir(directory_name) if fnmatch.fnmatch(file, file_pattern)]

        if wait and not files:
            if time.time() - start_time > timeout:
                print("Timeout reached while waiting for files.")
                break
            time.sleep(INTERVAL)
            continue

        else:
            break

    files = [file for file in os.listdir(directory_name) if fnmatch.fnmatch(file, file_pattern)]

    if files:
        # Extraer el número de ronda de cada file
        round_numbers = [int(re.search(r"round-(\d+)-weights\.npz", file).group(1)) for file in files]
        return max(round_numbers)
    
    else:
        return 0


def init_containers(federation, devices, data_type):
    local_run(f"docker-compose -f dockerfiles/{federation}.yml up --build -d", True)
    time.sleep(2)

    if federation == "remote-execution":
        update_clients()

        for i, (user, ip) in enumerate(devices.items(), start=1):
            create_network = "docker network create --driver bridge flwr-network"

            supernode_command = f"docker run --rm \
                --name supernode-{i} \
                -p 9094:9094 \
                --network flwr-network \
                --env DEVICE=supernode-{i} \
                --detach \
                flwr/supernode:1.15.1 \
                --insecure \
                --superlink 172.24.100.143:9092 \
                --node-config 'partition-id={i-1} num-partitions={len(devices)}  partition-type=\"{data_type}\"' \
                --clientappio-api-address 0.0.0.0:9094 \
                --isolation process"
            
            client_command = f"docker pull victorhidalgo/clientapp && \
                docker run --rm \
                --name client-{i} \
                --network flwr-network \
                --env DEVICE=supernode-{i} \
                --cpuset-cpus='1,2' \
                --detach \
                victorhidalgo/clientapp \
                --insecure \
                --clientappio-api-address supernode-{i}:9094"

            remote_run(create_network, ip, user)
            remote_run(supernode_command, ip, user)
            remote_run(client_command, ip, user)
  

def main():
    global handler_flag

    args = parser.parse_args()

    debug_mode = args.debug
    federations = args.federation
    sync_list = args.sync_clients
    data_list = args.data_list
    number_execution = args.number_execution

    event = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, event))

    writeConfig_cmd = 'python configWriter.py'
    local_run(writeConfig_cmd, False)

    try:
        config = toml.load(projectconf)
    except FileNotFoundError:
        print(f"El archivo {projectconf} no se encuentra. Asegúrate de que exista.")
        exit(1)

    strategies = ["FedAvg"]

    for data_type in data_list:
        for federation in federations:
            config['tempConfig']['federation'] = federation

            devices = config.get("devices", {})
            clients = config.get("clients", {})
            rounds = config['config']['rounds']

            config['config']['debug'] = debug_mode

            with open(projectconf, 'w') as f:
                toml.dump(config, f)

            if federation == 'local-execution':
                local_run(f"python local_execution_yaml.py -c {len(clients)} -t 1 -d {data_type}", True)

            for strategy in strategies:
                for sync_number in sync_list:
                    for act_exec in range(1, number_execution+1):   
                        init_containers(federation, devices, data_type) 

                        # Send an HTTP POST request to delete data series on the server
                        requests.post(url, params=params)

                        execution_name = f"sync{sync_number}_data{data_type}"

                        config['tempConfig']['strategy'] = strategy
                        config['tempConfig']['execution_name'] = execution_name
                        config['tempConfig']['num_exec'] = act_exec

                        if sync_number > len(clients):
                            print("Defaulting to asynchrony because m was higher than amount of clients")
                            sync_number = len(clients)

                        config['synchrony'] = sync_number

                        with open(projectconf, 'w') as f:
                            toml.dump(config, f)

                        last_round = get_last_round(config, strategy, act_exec, federation, execution_name)
                        step_rounds = rounds - last_round

                        while step_rounds > 0:
                            config['tempConfig']['step_rounds'] = step_rounds
                            config['tempConfig']['last_round'] = last_round

                            with open(projectconf, 'w') as f:
                                toml.dump(config, f)

                            local_run(f"flwr run . {federation}", False)
                            docker_logs = local_Popen(f"docker logs -f serverapp", True)

                            log_thread = threading.Thread(target=check_docker_logs, args=(event,federation,strategy,execution_name,act_exec,), daemon=True)
                            log_thread.start()

                            log_thread.join()

                            docker_logs.terminate()
                            docker_logs.wait()

                            # Exit if abrupt termination (SIGINT)
                            if event.is_set():
                                print("Ctrl-C detected, stopping everything...")
                                event.clear()

                                local_run(f"docker-compose -f dockerfiles/{federation}.yml down")

                                if federation == "remote-execution":
                                    for i, (user, ip) in enumerate(devices.items(), start=1):
                                        stop_remote_code(i, user, ip)

                                
                                if handler_flag:
                                    exit()

                            last_round = get_last_round(config, strategy, act_exec, federation, execution_name, True)
                            print(f"Inner last round {last_round}")
                            step_rounds = rounds - last_round

                        graphfl_cmd = f"python Analysis/timestamp_graph.py -f {federation} --strategy {strategy} -s {sync_number} -t {data_type} -n {act_exec}"
                        local_run(graphfl_cmd, True)

                        local_run(f"docker-compose -f dockerfiles/{federation}.yml down")

                        if federation == "remote-execution":
                            for i, (user, ip) in enumerate(devices.items(), start=1):
                                stop_remote_code(i, user, ip)

    
if __name__ == "__main__":
    main()
