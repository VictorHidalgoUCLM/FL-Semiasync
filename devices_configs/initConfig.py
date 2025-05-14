import subprocess
import sys
import os
import configparser
import yaml
import shutil

def update_prometheus(file_yaml, data, device_number):
    # Read YAML file
    with open(file_yaml, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Update values
    for config in yaml_data['scrape_configs']:
        config['static_configs'][device_number]['targets'] = [data['target']]
        config['static_configs'][device_number]['labels'].update(data)

    # Write updated YAML file
    with open(file_yaml, 'w') as file:
        yaml.dump(yaml_data, file)


config_path = 'initConfig.py'

config = configparser.ConfigParser()
config.read(config_path)

devices = dict(config['configDevices'])

data_to_save = {
    "country": "",
    "environment": "",
    "client": "",
    "hostname": "",
}

# Define arrays for remote devices
remote_users = []
remote_data = {}

remote_dirs = []
remote_dirs_env = []

for key, value in config['configDevices'].items():
    remote_users.append(key)
    remote_data[key] = eval(value)

    remote_dirs.append("/home/" + key)
    remote_dirs_env.append("/home/" + key + "/slave")

# Local directory to transfer
local_data = "data"
local_env = ".env"

max_client_id = len(remote_users) - 1
num_clients = len(remote_users)
data_type = 1

# Read same_client_id from command line argument
try:
    same_client_id = sys.argv[1].lower() == 'true'
except IndexError:
    same_client_id = False

unique_client_id = 0

# Loop to execute the script for each remote device
for i in range(max_client_id + 1):
    if same_client_id:
        client_id = unique_client_id
    else:
        client_id = i

    print(f"Value of client_id: {client_id}")

    remote_user = remote_users[i]
    remote_host = remote_data[remote_user][0]
    remote_dir = remote_dirs[i]
    remote_dir_env = remote_dirs_env[i]

    data_to_save["target"] = f"{remote_data[remote_user][0]}" + ":9090"
    data_to_save["country"] = remote_data[remote_user][1]
    data_to_save["environment"] = remote_data[remote_user][2]
    data_to_save["client"] = remote_data[remote_user][3]
    data_to_save["hostname"] = remote_user

    with open(local_env, "w") as env_file:
        for key, value in data_to_save.items():
            env_file.write(f"{key}={value}\n")

    update_prometheus('master/prometheus.yml', data_to_save, i)

    # Copy directory using scp
    scp_command1 = f"scp -r {local_data} {remote_user}@{remote_host}:{remote_dir}"
    scp_process = subprocess.Popen(scp_command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    scp_process.communicate()

    # Check the exit status of scp
    if scp_process.returncode == 0:
        print(f"Directory transferred successfully to {remote_host}!")
    else:
        print(f"Error transferring directory to {remote_host}.")

    # Copy directory using scp
    scp_command2 = f"scp -r {local_env} {remote_user}@{remote_host}:{remote_dir_env}"
    scp_process = subprocess.Popen(scp_command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    scp_process.communicate()

    # Check the exit status of scp
    if scp_process.returncode == 0:
        print(f"Env transferred successfully to {remote_host}!")
 
        comm = "cd slave && docker compose down && docker compose up -d"
        subprocess.Popen(comm, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)

    else:
        print(f"Error transferring env to {remote_host}.")

    shutil.rmtree(local_data)
    os.remove(local_env)

comm = "cd master && docker compose down && docker compose up -d"

try:
    # Ejecutar el comando y capturar la salida y los errores
    proceso = subprocess.Popen(comm, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

except Exception as e:
    print(f"Error al ejecutar el comando '{comm}': {e}")