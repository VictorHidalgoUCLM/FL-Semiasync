import toml
import sys

# Variables needed later
file_path = 'projectconf.toml'

default_sliding_window = 1024
default_new_rounds = 100
default_new_server = 1
default_new_dataset = "uoft-cs/cifar10"

if len(sys.argv) > 1:
    try:
        new_value = int(sys.argv[1])
        new_rounds = int(sys.argv[2])
        new_server = int(sys.argv[3])
        new_dataset = str(sys.argv[4])
    except ValueError:
        print("Error, no es un número entero. Usando default (new_value)...")
        new_value = default_sliding_window
        new_rounds = default_new_rounds
        new_server = default_new_server
        new_dataset = default_new_dataset
    
else:
    new_value = default_sliding_window
    new_rounds = default_new_rounds
    new_server = default_new_server
    new_dataset = default_new_dataset

# Each dictionary is one kind of configuration for the project
paths = {
    'checkpoint': '/app/results/{federation}/{strategy}/{sub_execution}/checkpoints/{num_exec}',
    'metrics': '/app/results/{federation}/{strategy}/{sub_execution}/metrics',
    'logs': '/app/results/{federation}/{strategy}/{sub_execution}/logs',
    'localCheckpoint': '{federation}/results/{strategy}/{sub_execution}/checkpoints/{num_exec}',
    'localLog': '{federation}/results/{strategy}/{sub_execution}/logs/log_{num_exec}.csv',
    'localTimestamp': '{federation}/results/{strategy}/{sub_execution}/logs/timestamp_{num_exec}.csv',
}

config = {
    'fraction_fit': 1,
    'fraction_evaluate': 0,
    'min_fit_clients': 8,
    'min_evaluate_clients': 8,
    'min_available_clients': 8,
    'evaluate_fn': "evaluate_server",
    'on_fit_config_fn': "None",
    'on_evaluate_config_fn': "None",
    'accept_failures': True,
    'initial_parameters': "server_side",
    'fit_metrics_aggregation_fn': "fit_weighted_average",
    'evaluate_metrics_aggregation_fn': "evaluate_weighted_average",
    'rounds': new_rounds,
    'dataset_name': new_dataset,
    'server_type': new_server,
    'inplace': True,
}

devices = {
    'raspberrypi1': 'raspberrypi1',
    'raspberry4': 'raspberry4',
    'raspberry3': 'raspberry3',
    'raspberry6': 'raspberry6',
    'raspberry7': 'raspberry7',
    'raspberry2': 'raspberry2',
    'raspberry5': 'raspberry5',
    'raspberry8': 'raspberry8',
}

clients = {
    'raspberrypi1': [1, 32, new_value],
    'raspberry4': [1, 32, new_value],
    'raspberry3': [1, 32, new_value],
    'raspberry6': [1, 32, new_value],
    'raspberry7': [1, 32, new_value],
    'raspberry2': [1, 32, new_value],
    'raspberry5': [1, 32, new_value],
    'raspberry8': [1, 32, new_value],
}

prometheus_conf = {
    'prometheus_url': 'http://master_prometheus:9090',
    'sleep_time': 5,
}

local_execution = {
    'devices': ["victorPC"],
}

remote_execution = {
    'devices': ["raspberrypi1", "raspberry4", "raspberry3", "raspberry6", "raspberry7"],
}

names = {
    'supernode-1': [1, 32, new_value],
    'supernode-2': [1, 32, new_value],
    'supernode-3': [1, 32, new_value],
    'supernode-4': [1, 32, new_value],
    'supernode-5': [1, 32, new_value],
    'supernode-6': [1, 32, new_value],
    'supernode-7': [1, 32, new_value],
    'supernode-8': [1, 32, new_value],
    'clientapps': ["client-1", "client-2", "client-3", "client-4", "client-5", "client-6", "client-7", "client-8",]
}

fedProx = {
    'proximal_mu': 0.1,
}

conf = {
    'paths': {},
    'config': {},
    'devices': {},
    'clients': {},
    'prometheus_conf': {},
    'local-execution': {},
    'remote-execution': {},
    'tempConfig': {},
    'names': {},
    'fedProx': {},
}

# Write the new updated configuration data
conf['paths'].update(paths)
conf['config'].update(config)
conf['devices'].update(devices)
conf['clients'].update(clients)
conf['prometheus_conf'].update(prometheus_conf)
conf['local-execution'].update(local_execution)
conf['remote-execution'].update(remote_execution)
conf['names'].update(names)
conf['fedProx'].update(fedProx)


# Write the configured data
with open(file_path, 'w') as configfile:
    toml.dump(conf, configfile)

print(f"{file_path} succesfully updated.")
