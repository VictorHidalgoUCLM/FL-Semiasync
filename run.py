import argparse
import fnmatch
import os
import re
import signal
import subprocess
import threading
import time
import random
import sys

import requests
import toml
from itertools import product

projectconf = 'projectconf.toml'
handler_flag = False
INTERVAL = 10

PERFILES_DIR = "profiles"
CICLOS = None

# Configuration to delete data series on the server
url = 'http://localhost:9090/api/v1/admin/tsdb/delete_series'
params = {'match[]': '{job="cadvisor"}'}

parser = argparse.ArgumentParser(description="Main program for executing FL on Flwr on different configurations.")
parser.add_argument(
    "-f", "--federation", 
    type=str, 
    nargs='+',
    choices=["local-execution", "remote-execution"], 
    help="Execution modes: 'local-execution' or 'remote-execution'.",
    default=["local-execution"]
)

parser.add_argument(
    '-m', '--sync-clients',
    type=int,
    nargs='+',
    help="Number of clients required for synchronization in each training round.",
    default=[8]
)

parser.add_argument(
    '-t', '--data-list',
    type=str,
    nargs='+',
    help="List of dataset types to run (e.g. iid, non-iid).",
    default=['iid']
)

parser.add_argument(
    '-n', '--num-executions',
    type=int,
    help="Number of repeated executions of the experiment.",
    default=1
)

parser.add_argument(
    '-r', '--rounds',
    type=int,
    help="Number of training rounds to run.",
    default=10
)

parser.add_argument(
    '-S', '--slowclients',
    type=int,
    nargs='+',
    help="List of client IDs or counts to be considered slow.",
    default=[0]
)

parser.add_argument(
    '-M', '--medclients',
    type=int,
    nargs='+',
    help="List of client IDs or counts to be considered medium.",
    default=[0]
)

parser.add_argument(
    '-a', '--asynchrony',
    type=int,
    nargs='+',
    help="List of semiasynchrony tipes (1 = semiasync, 2 = modded semiasync).",
    default=[1]
)

parser.add_argument(
    '-d', '--dataset',
    type=str,
    default="uoft-cs/cifar10"
)

parser.add_argument(
    '-s', '--strategies',
    type=str,
    nargs='+',
    help="List of Federated strategies.",
    default=["FedAvg"]
)

parser.add_argument(
    '-b', '--betas',
    type=float,
    nargs='+',
    help="List of Federated betas.",
    default=[0.9]
)

parser.add_argument(
    '-g', '--gammas',
    type=float,
    nargs='+',
    help="List of Federated gammas.",
    default=[0.9]
)

parser.add_argument(
    '-i',
    type=float,
    nargs='+',
    help="List of alphas considered for non-iid data distribution.",
    default=[0.5]
)

parser.add_argument(
    '--debug',
    action='store_true',
    help="Enable debug mode."
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
    """
    Monitors the logs of the Docker container 'serverapp' for a specific pattern
    that indicates the end of a training process.

    The function builds a log pattern in the form:
        "Training {federation} {strategy} {execution_name} {num_exec} ended"
    and periodically checks the last lines of the container logs until either:
        1. The pattern is found (meaning the training has finished), or
        2. The external event 'event' is set.
    """
    pattern_found_event = threading.Event()
    PATTERN = f"Training {federation} {strategy} {execution_name} {num_exec} ended" # Pattern to be search

    while not pattern_found_event.is_set() and not event.is_set():
        try:
            # Executes `docker logs` command and gets last 4 lines
            result = subprocess.run(
                ["docker", "logs", "--tail", "4", "serverapp"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Checks for pattern
            if PATTERN in result.stdout.strip():
                pattern_found_event.set()

        except subprocess.CalledProcessError as e:
            print(f"Error while executing...: {e}")

        # Waits the predefined interval between while loop
        time.sleep(INTERVAL)


def local_run(command, show_output=False):
    """
    Executes a shell command locally, with optional output display.

    The command is split into parts and executed using subprocess. By default,
    the command output (stdout and stderr) is suppressed, but it can be shown
    if `show_output=True` is provided.
    """
    try:
        command_parts = command.split()
        print(f"\nExecuting {command}...\n")

        if show_output:
            subprocess.run(command_parts, check=True)
        else:
            subprocess.run(command_parts, stdout=sys.stdout, stderr=sys.stderr, check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error when executing Docker command: {e}")
    

def local_Popen(command, show_output=False, output_file='salida.txt'):
    """
    Launches a local process using subprocess.Popen, with optional output display.

    The command is split into parts and executed asynchronously. Unlike `subprocess.run`,
    this function does not block execution and instead returns the Popen process object.
    """
    try:
        command_parts = command.split()
        print(f"\nExecuting {command}...\n")

        if show_output:
            f = open(output_file, "w")

            process = subprocess.Popen(command_parts, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            import threading
            def log_writer():
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
            
            threading.Thread(target=log_writer, daemon=True).start()

        else:
            process = subprocess.Popen(command_parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError as e:
        print(f"Error when executing Docker command: {e}")

    return process


def remote_run(command, ip, user):
    """
    Executes a shell command on a remote machine over SSH.

    The function builds an SSH command in the form:
        ssh user@ip <command>
    and runs it locally using subprocess. By default, both stdout and stderr 
    are suppressed.
    """
    try:
        ssh_command = ["ssh", f"{user}@{ip}", command]
        print(f"Ejecutando comando remoto: {ssh_command}")
        subprocess.run(ssh_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    except subprocess.CalledProcessError as e:
        pass


def stop_remote_code(id, user, ip):
    """
    Stops remote Docker containers and removes the associated network on a remote host.

    Specifically, it stops the containers:
        - supernode-{id}
        - client-{id}
    and removes the Docker network 'flwr-network' by executing SSH commands 
    on the specified remote machine.
    """
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
    """
    Builds, tags, and pushes the Docker image for the client application.

    Steps performed:
    1. Builds the Docker image using the Dockerfile 'dockerfiles/clientapp.Dockerfile'
       targeting the ARM64 platform and tags it as 'flwr_clientapp:latest'.
    2. Tags the local image with the repository 'victorhidalgo/clientapp'.
    3. Pushes the tagged image to the Docker registry.
    """
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
    """
    Initializes and starts Docker containers for the federated learning setup.

    Depending on the federation type, this function either launches local containers
    or deploys them on remote machines.

    Steps:
    1. Uses `docker compose` to start the containers defined in the corresponding
       docker-compose YAML file for the given federation.
    2. If `federation` is "remote-execution":
        - Updates client Docker images using `update_clients()`.
        - Iterates over the provided `devices` dictionary and for each device:
            a) Creates a Docker network 'flwr-network' remotely.
            b) Runs a supernode container with configuration specific to the partition.
            c) Pulls and runs a client container attached to the same network.
    """
    local_run(f"docker compose -f dockerfiles/{federation}.yml up --build -d", True)
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
                flwr/supernode:1.18.0 \
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


def generar_perfil_cpus_interpolado(
    duracion_total,
    num_cambios,
    cpu_max,
    cpu_min,
    min_primer_intervalo=300
):
    """
    Genera un perfil de CPU interpolando entre cpu_max y cpu_min.
    El primer tramo siempre usa cpu_max.
    """
    if num_cambios < 1:
        raise ValueError("num_cambios debe ser >= 1")

    tiempo_restante = duracion_total - min_primer_intervalo
    if tiempo_restante <= 0:
        raise ValueError("duracion_total demasiado pequeña")

    cambios_restantes = num_cambios - 1

    if cambios_restantes > 0:
        cortes = [
            i * tiempo_restante / cambios_restantes
            for i in range(1, cambios_restantes)
        ]

        intervalos = (
            [cortes[0]] +
            [cortes[i] - cortes[i - 1] for i in range(1, len(cortes))] +
            [tiempo_restante - cortes[-1]]
        )
    else:
        intervalos = []

    intervalos = [min_primer_intervalo] + intervalos

    perfil = []
    tiempo_acumulado = 0.0

    # Primer tramo: CPU max
    perfil.append({
        "start": tiempo_acumulado,
        "duration": intervalos[0],
        "cpu": cpu_max
    })
    tiempo_acumulado += intervalos[0]

    # Interpolación lineal
    for i, intervalo in enumerate(intervalos[1:], start=1):
        alpha = i / (num_cambios - 1)  # 0 → 1
        cpu = cpu_max - alpha * (cpu_max - cpu_min)

        perfil.append({
            "start": tiempo_acumulado,
            "duration": intervalo,
            "cpu": round(cpu, 3)
        })
        tiempo_acumulado += intervalo

    return perfil

def ejecutar_perfil(container_name, perfil, ciclos=1):
    """
    Ejecuta docker update según el perfil, repitiéndolo 'ciclos' veces.
    """
    for ciclo in range(ciclos):
        print(f"[{container_name}] Ciclo {ciclo+1}/{ciclos}")

        for i, event in enumerate(perfil):
            cpu = event["cpu"]
            dur = event["duration"]

            print(f"[{container_name}] Cambio {i+1}/{len(perfil)} -> CPU={cpu} durante {dur:.2f}s")

            os.system(f"docker update --cpus={cpu} {container_name}")

            time.sleep(dur)

    print(f"[{container_name}] Perfil completado.")

def lanzar_todos_perfiles_generados(clientes, duracion, cambios, cpu_max, cpu_min, min_primer_intervalo, ciclos=1, seed=42):
    """
    Genera y lanza perfiles aleatorios para cada cliente en hilos
    """
    random.seed(seed)
    hilos = []

    for client_id in clientes:
        contenedor = f"client-{client_id}"

        # Generar perfil aleatorio
        perfil = generar_perfil_cpus_interpolado(
            duracion_total=duracion,
            num_cambios=cambios,
            cpu_max=cpu_max,
            cpu_min=cpu_min,
            min_primer_intervalo=min_primer_intervalo
        )

        # Lanzar hilo
        t = threading.Thread(target=ejecutar_perfil, args=(contenedor, perfil, ciclos))
        t.daemon = True
        t.start()
        hilos.append(t)

        print(f"[INFO] Lanzado perfil de {contenedor}")
    return hilos


def load_and_update_config(projectconf, server_type, dataset, federation, local_run, rounds):
    # 1. Escribir config temporal
    writeConfig_cmd = f'python configWriter.py 1 {rounds} {server_type} {dataset}'
    local_run(writeConfig_cmd, True)

    # 2. Cargar config
    try:
        config = toml.load(projectconf)
    except FileNotFoundError:
        print(f"Could not find {projectconf} file.")
        return None

    # 3. Actualizar campos necesarios
    config['tempConfig']['federation'] = federation
    devices = config.get("devices", {})
    clients = {
        k: v for k, v in config.get("names", {}).items() if k != "clientapps"
    }
    rounds = config['config']['rounds']

    # 4. Guardar archivo actualizado
    with open(projectconf, 'w') as f:
        toml.dump(config, f)

    # 5. Devolver lo útil
    return config, devices, clients, rounds


def update_temp_config(config, strategy, execution_name, act_exec, clients, inplace, sync_number, beta=0.9, gamma=0.9, debug=False):
    temp = config.setdefault('tempConfig', {})
    temp['strategy'] = strategy
    temp['execution_name'] = execution_name
    temp['num_exec'] = act_exec
    temp['beta'] = beta
    temp['gamma'] = gamma
    temp['debug'] = debug

    config["config"]["inplace"] = inplace

    if sync_number > len(clients):
        print("Defaulting to asynchrony because m was higher than amount of clients")
        sync_number = len(clients)

    config['synchrony'] = sync_number

    with open(projectconf, 'w') as f:
        toml.dump(config, f)


def main():
    global handler_flag

    args = parser.parse_args()  # Get parser

    # Get all parse options
    federations = args.federation
    sync_list = args.sync_clients
    data_list = args.data_list
    num_executions = args.num_executions
    rounds = args.rounds
    slowclients = args.slowclients
    medclients = args.medclients
    asynchronies = args.asynchrony
    dataset = args.dataset
    strategies = args.strategies
    betas = args.betas
    gammas = args.gammas
    alphas = args.i
    debug = args.debug

    event = threading.Event()
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, event))

    # Iterate through data_list, federations, window_size, slowclients, heterogeneities
    for server_type, data_type, federation, alpha in product(
        asynchronies, data_list, federations, alphas
    ):
        for slow_client, med_client in zip(slowclients, medclients):
            config, devices, clients, rounds = load_and_update_config(
                projectconf, server_type, dataset, federation, local_run, rounds)

            if federation == 'local-execution':
                cmd = (
                    f"python local_execution_yaml.py "
                    f"-c {len(clients)} "
                    f"-t 1 "
                    f"-d {data_type} "
                    f"-s {slow_client} "
                    f"-m {med_client} "
                    f"-n {dataset} "
                    f"-i {alpha} "
                )

                local_run(cmd, True)

            if data_type == "noniid":
                data_type = f"{data_type}{alpha}"

            # Iterate through strategies, sync_list and num_executions (1 to num_executions + 1)
            for strategy, sync_number, beta, gamma, act_exec in product(
                strategies, sync_list, betas, gammas, range(1, num_executions + 1)):

                local_run("docker compose -f devices_configs/master/docker-compose.yml up -d", True)
                init_containers(federation, devices, data_type) 

                if strategy == "FedMOpt":
                    execution_name = f"{data_type}/slow{slow_client}_med{med_client}/beta{beta}/gamma{gamma}"
                else:
                    execution_name = f"{data_type}/sync{sync_number}/slow{slow_client}_med{med_client}/beta{beta}/gamma{gamma}"

                inplace = False
                update_temp_config(config, strategy, execution_name, act_exec, clients, inplace, sync_number, beta, gamma, debug)

                last_round = get_last_round(config, strategy, act_exec, federation, execution_name)
                step_rounds = rounds - last_round

                while step_rounds > 0:
                    config['tempConfig']['step_rounds'] = step_rounds
                    config['tempConfig']['last_round'] = last_round

                    with open(projectconf, 'w') as f:
                        toml.dump(config, f)

                    local_run(f"flwr run . {federation}", False)

                    # In case we need to generate false CPU usage
                    """lanzar_todos_perfiles_generados(
                        clientes=[i+1 for i in range(slow_client)],
                        duracion=1000, 
                        cambios=15,
                        cpu_min=0.2,
                        cpu_max=1.0,
                        min_primer_intervalo=200
                    )
"""
                    docker_logs = local_Popen(
                        f"docker logs -f serverapp",
                        True,
                        f"{federation}/results/{strategy}/{execution_name}/output_log_{act_exec}.txt"
                    )

                    log_thread = threading.Thread(
                        target=check_docker_logs,
                        args=(event,federation,strategy,execution_name,act_exec,), 
                        daemon=True
                    )

                    log_thread.start()

                    log_thread.join()

                    docker_logs.terminate()
                    docker_logs.wait()

                    f.close()
                    
                    # Exit if abrupt termination (SIGINT)
                    if event.is_set():
                        print("Ctrl-C detected, stopping everything...")
                        event.clear()

                        f.close()

                        local_run(f"docker compose -f dockerfiles/{federation}.yml down")
                        # Send an HTTP POST request to delete data series on the server
                        requests.post(url, params=params)
                        local_run("docker compose -f devices_configs/master/docker-compose.yml down")

                        if federation == "remote-execution":
                            for i, (user, ip) in enumerate(devices.items(), start=1):
                                stop_remote_code(i, user, ip)

                        
                        if handler_flag:
                            exit()

                    last_round = get_last_round(config, strategy, act_exec, federation, execution_name, True)
                    print(f"Inner last round {last_round}")
                    step_rounds = rounds - last_round

                local_run(f"docker compose -f dockerfiles/{federation}.yml down")
                # Send an HTTP POST request to delete data series on the server
                requests.post(url, params=params)
                local_run("docker compose -f devices_configs/master/docker-compose.yml down")

                if federation == "remote-execution":
                    for i, (user, ip) in enumerate(devices.items(), start=1):
                        stop_remote_code(i, user, ip)

    
if __name__ == "__main__":
    main()
