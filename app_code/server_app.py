import os
import toml

# Flower imports
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from .serverapp import ServerAppCustom

# Typing imports
from typing import List, Tuple

# Custom server import
from .server import CustomServer

# Task-related imports
from app_code.task import Net, get_weights

# Strategy imports
from app_code.strategies.FedAvg import FedAvgCustom
from app_code.strategies.FedProx import FedProxCustom
from app_code.strategies.FedYogi import FedYogiCustom
from app_code.strategies.FedAdam import FedAdamCustom
from app_code.strategies.FedAdagrad import FedAdagradCustom
from app_code.strategies.FedTrimmedAvg import FedTrimmedAvgCustom
from app_code.strategies.QFedAvg import QFedAvgCustom
from app_code.strategies.FedMedian import FedMedianCustom

projectconf = toml.load(os.environ.get('CONFIG_PATH'))

num_exec = projectconf['tempConfig']['num_exec']
strategy_name = projectconf['tempConfig']['strategy']

# Create strategy based on the strategy selected in the configuration file
def fedAvg(configurations):
    strategy = FedAvgCustom(**configurations)
    return strategy

def fedProx(configurations):
    strategy = FedProxCustom(**configurations,
                            parameters_fit= {
                                'proximal_mu': projectconf['fedProx']['proximal_mu'],
                                }
                            )
    return strategy

def fedYogi(configurations):
    strategy = FedYogiCustom(**configurations)
    return strategy

def fedAdam(configurations):
    strategy = FedAdamCustom(**configurations)
    return strategy

def fedAdagrad(configurations):
    strategy = FedAdagradCustom(**configurations)
    return strategy

def fedTrimmedAvg(configurations):
    strategy = FedTrimmedAvgCustom(**configurations)
    return strategy

def QFedAvg(configurations):
    strategy = QFedAvgCustom(**configurations)
    return strategy

def fedMedian(configurations):
    strategy = FedMedianCustom(**configurations)
    return strategy

# Default function if the strategy is not selected correctly
def default():
    return "Nothing selected"

# Map strategy names to functions
switch = {
    "FedAvg": fedAvg,
    "FedProx": fedProx,
    "FedYogi": fedYogi,
    "FedAdam": fedAdam,
    "FedAdagrad": fedAdagrad,
    "FedTrimmedAvg": fedTrimmedAvg,
    "QFedAvg": QFedAvg,
    "FedMedian": fedMedian,
}


# Function for calculating the weighted average metric
def fit_weighted_average(
        metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    losses_distributed = [num_examples * m["loss_distributed"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples), "loss_distributed": sum(losses_distributed) / sum(examples), "recall": sum(recalls) / sum(examples), "precision": sum(precisions) / sum(examples), "f1": sum(f1s) / sum(examples)}

# Function for calculating the weighted average metric
def evaluate_weighted_average(
        metrics: List[Tuple[int, fl.common.Metrics]]) -> fl.common.Metrics:

    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples), "recall": sum(recalls) / sum(examples), "precision": sum(precisions) / sum(examples), "f1": sum(f1s) / sum(examples)}


def server_side_parameters():
    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    return parameters


# Dictionary mapping names to functions
functions_dict = {
    'fit_weighted_average': fit_weighted_average,
    'evaluate_weighted_average': evaluate_weighted_average,
    'server_side': server_side_parameters,
    'None': None,
}

configurations = {
    "fraction_fit": projectconf['config']["fraction_fit"],
    "fraction_evaluate": projectconf['config']["fraction_evaluate"],
    "min_fit_clients": projectconf['config']["min_fit_clients"],
    "min_evaluate_clients": projectconf['config']["min_evaluate_clients"],
    "min_available_clients": projectconf['config']["min_available_clients"],
    "evaluate_fn": functions_dict.get(projectconf['config']["evaluate_fn"], None),
    "on_fit_config_fn": functions_dict.get(projectconf['config']["on_fit_config_fn"], None),
    "on_evaluate_config_fn": functions_dict.get(projectconf['config']["on_evaluate_config_fn"], None),
    "accept_failures": projectconf['config']["accept_failures"],
    "initial_parameters": functions_dict.get(projectconf['config']["initial_parameters"], None),
    "fit_metrics_aggregation_fn": functions_dict.get(projectconf['config']["fit_metrics_aggregation_fn"], None),
    "evaluate_metrics_aggregation_fn": functions_dict.get(projectconf['config']["evaluate_metrics_aggregation_fn"], None),
    "num_exec": num_exec,
    "strategy_name": strategy_name,
    "debug": projectconf['config']['debug']
}

# Map values from configuration to functions based on the dictionary
for key, value in configurations.items():
    if value in functions_dict:
        configurations[key] = functions_dict[value]

configurations["initial_parameters"] = configurations["initial_parameters"]()

# Select strategy based on the name provided in the configuration file
selected_strategy = switch.get(strategy_name, default)
strategy = selected_strategy(configurations)

def server_fn(context: Context):
    # Read from config
    num_rounds = projectconf['tempConfig']['step_rounds']
    offset = projectconf['tempConfig']['last_round']

    strategy.set_round_offset(offset)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, server=CustomServer(client_manager=SimpleClientManager(), strategy=strategy))

# Create ServerApp
app = ServerAppCustom(server_fn=server_fn)
