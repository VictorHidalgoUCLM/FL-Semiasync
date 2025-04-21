import toml

# Variables needed later
file_path = 'projectconf.toml'

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
    'fraction_evaluate': 1,
    'min_fit_clients': 5,
    'min_evaluate_clients': 5,
    'min_available_clients': 5,
    'evaluate_fn': "None",
    'on_fit_config_fn': "None",
    'on_evaluate_config_fn': "None",
    'accept_failures': False,
    'initial_parameters': "server_side",
    'fit_metrics_aggregation_fn': "fit_weighted_average",
    'evaluate_metrics_aggregation_fn': "evaluate_weighted_average",
    'rounds': 50,
}

devices = {
    'raspberrypi1': 'raspberrypi1',
    'raspberry4': 'raspberry4',
    'raspberry3': 'raspberry3',
    'raspberry6': 'raspberry6',
    'raspberry7': 'raspberry7',
}

clients = {
    'raspberrypi1': [1, 32, 1024],
    'raspberry4': [1, 32, 1024],
    'raspberry3': [1, 32, 1024],
    'raspberry6': [1, 32, 1024],
    'raspberry7': [1, 32, 1024],
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
    'supernode-1': [1, 32, 1024],
    'supernode-2': [1, 32, 1024],
    'supernode-3': [1, 32, 1024],
    'supernode-4': [1, 32, 1024],
    'supernode-5': [1, 32, 1024],
    'clientapps': ["client-1", "client-2", "client-3", "client-4", "client-5",]
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
