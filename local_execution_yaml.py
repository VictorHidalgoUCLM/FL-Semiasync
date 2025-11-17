import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--clients", 
    type=int, 
    help="Quantity of clients",
    required=True
)

parser.add_argument(
    "-s",
    "--slowclients", 
    type=int, 
    help="Quantity of slow clients",
    required=True
)

parser.add_argument(
    "-H",
    "--heterogeneity", 
    type=str, 
    help="Heterogeneity of clients",
    required=True
)

parser.add_argument(
    "-t",
    "--threads", 
    type=int, 
    help="Threads per client",
    required=True
)

parser.add_argument(
    "-d",
    "--data_type", 
    type=str, 
    help="Data types: iid | non-iid",
    required=True
)

parser.add_argument(
    "-n",
    "--name_dataset", 
    type=str, 
    help="Data types: uoft-cs/cifar10 | ylecun/mnist",
    default="uoft-cs/cifar10"
)

args = parser.parse_args()

n = args.clients
cpus = args.threads
data_type = args.data_type
heterogeneity = args.heterogeneity
slowclients = args.slowclients
dataset_name = args.name_dataset

superlink = {
    'image': 'flwr/superlink:1.18.0',
    'container_name': 'superlink',
    'ports': ['9091:9091', '9092:9092', '9093:9093'],
    'networks': ['master_default'],
    'command': ['--insecure', '--isolation', 'process'],
    'cpuset': f'{n}',
    'mem_limit': '2g'
}

serverapp = {
    'build': {
        'context': '..',
        'dockerfile': 'dockerfiles/serverapp.Dockerfile'
    },
    'container_name': 'serverapp',
    'networks': ['master_default'],
    'command': ['--insecure', '--serverappio-api-address', 'superlink:9091'],
    'depends_on': ['superlink'],
    'volumes': [
        '../local-execution/results:/app/results/local-execution:rw',
        '../projectconf.toml:/app/projectconf.toml:rw'
    ],
    'cpuset': f'{n+1}',
    'mem_limit': '2g' # 2 GB RAM memory limit
}

services = {
    'superlink': superlink,
    'serverapp': serverapp
}

half = slowclients // 2

for i in range(1, n+1):
    supernode_name = f'supernode-{i}'
    client_name = f'client-{i}'

    supernode_config = {
        'image': 'flwr/supernode:1.18.0',
        'container_name': supernode_name,
        'ports': [f'{9093 + i}:{9093 + i}'],
        'networks': ['master_default'],
        'command': [
            '--insecure', '--superlink', 'superlink:9092',
            '--node-config', f'partition-id={i-1} num-partitions={n} partition-type="{data_type}" dataset-name="{dataset_name}"',
            '--clientappio-api-address', f'0.0.0.0:{9093 + i}', '--isolation', 'process'
        ],
        'depends_on': ['superlink'],
        'environment': [f'DEVICE={supernode_name}'],
        'cpuset': f'{i-1}',
        'mem_limit': '1g' # 1 GB RAM memory limit
    }

    cpuset_list = [str(10 + j + (i - 1) * cpus) for j in range(cpus)]
    cpuset_string = ','.join(cpuset_list)

    if heterogeneity == "heterogeneous":
        #if i <= half:
            #cpus_limit = 0.33
        #el
        if i <= slowclients:
            cpus_limit = 0.33
        else:
            cpus_limit = cpus

    elif heterogeneity == "homogeneous":
        cpus_limit = cpus

    print(cpus_limit)   

    client_config = {
        'build': {
            'context': '..',
            'dockerfile': 'dockerfiles/clientapp.Dockerfile'
        },
        'container_name': client_name,
        'networks': ['master_default'],
        'command': ['--insecure', '--clientappio-api-address', f'{supernode_name}:{9093 + i}'],
        'depends_on': [supernode_name],
        'environment': [f'DEVICE={supernode_name}'],
        'mem_limit': '2g',
        'cpuset': cpuset_string,
        'cpus': cpus_limit
    }

    # Add the configurations to the services dictionary
    services[supernode_name] = supernode_config
    services[client_name] = client_config

# Definir la estructura del archivo YAML
data = {
    'version': '3.8',
    'services': services,
    'networks': {
        'master_default': {
            'external': True
        }
    }
}

# Escribir la estructura en un archivo YAML
with open('dockerfiles/local-execution.yml', 'w') as file:
    yaml.dump(data, file, sort_keys=False)

print("Archivo docker-compose.yml creado exitosamente.")
