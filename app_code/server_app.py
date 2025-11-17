import os
from typing import List, Tuple
import fnmatch
import re
import numpy as np
import csv

import flwr as fl
import toml
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager

from .server_semiasync import CustomServerSemiasync
from .server_modsemiasync import CustomServerModSemiasync
from .serverapp import ServerAppCustom
from app_code.task import Net, get_weights
from app_code.strategies.FedAdam import FedAdamCustom
from app_code.strategies.FedAdagrad import FedAdagradCustom
from app_code.strategies.FedAvg import FedAvgCustom
from app_code.strategies.FedMedian import FedMedianCustom
from app_code.strategies.FedProx import FedProxCustom
from app_code.strategies.FedTrimmedAvg import FedTrimmedAvgCustom
from app_code.strategies.FedYogi import FedYogiCustom
from app_code.strategies.QFedAvg import QFedAvgCustom
from app_code.strategies.FedMOpt import FedMOpt

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from flwr.common.typing import NDArrays, Scalar

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Subset


projectconf = toml.load(os.environ.get("CONFIG_PATH"))

num_exec = projectconf["tempConfig"]["num_exec"]
strategy_name = projectconf["tempConfig"]["strategy"]

# Create strategy based on the strategy selected in the configuration file
def create_fedAvg(configurations):
    """Create FedAvg based on configurations parameter."""
    strategy = FedAvgCustom(**configurations)
    return strategy


def create_fedProx(configurations):
    """Create FedProx based on configurations parameter. Includes proximal_mu."""
    strategy = FedProxCustom(
        **configurations,
        parameters_fit={
            "proximal_mu": projectconf["fedProx"]["proximal_mu"],
        }
    )
    return strategy


def create_fedYogi(configurations):
    """Create FedYogi based on configurations parameter."""
    strategy = FedYogiCustom(**configurations)
    return strategy


def create_fedAdam(configurations):
    """Create FedYogi based on configurations parameter."""
    strategy = FedAdamCustom(**configurations)
    return strategy


def create_fedAdagrad(configurations):
    """Create FedAdagrad based on configurations parameter."""
    strategy = FedAdagradCustom(**configurations)
    return strategy


def create_fedTrimmedAvg(configurations):
    """Create FedTrimmedAvg based on configurations parameter."""
    strategy = FedTrimmedAvgCustom(**configurations)
    return strategy


def create_QFedAvg(configurations):
    """Create QFedAvg based on configurations parameter."""
    strategy = QFedAvgCustom(**configurations)
    return strategy


def create_fedMedian(configurations):
    """Create FedMedian based on configurations parameter."""
    strategy = FedMedianCustom(**configurations)
    return strategy

def create_fedMOpt(configurations):
    """Create FedMOpt based on configurations parameter."""
    strategy = FedMOpt(**configurations)
    return strategy


def default_strategy():
    """Default function if strategy is not selected correctly."""
    return "Nothing selected"


def get_strategy_mapping():
    """Returns a dictionary with strategy name as key and function as value."""
    return {
        "FedAvg": create_fedAvg,
        "FedProx": create_fedProx,
        "FedYogi": create_fedYogi,
        "FedAdam": create_fedAdam,
        "FedAdagrad": create_fedAdagrad,
        "FedTrimmedAvg": create_fedTrimmedAvg,
        "QFedAvg": create_QFedAvg,
        "FedMedian": create_fedMedian,
        "FedMOpt": create_fedMOpt,
    }


def fit_weighted_average(
    metrics: List[Tuple[int, fl.common.Metrics]],
) -> fl.common.Metrics:
    """Function for calculating weighted average metrics on fit."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "f1": sum(f1s) / sum(examples),
    }


# Function for calculating the weighted average metric
def evaluate_weighted_average(
    metrics: List[Tuple[int, fl.common.Metrics]],
) -> fl.common.Metrics:
    """Function for calculating weighted average metrics on evaluate"""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    recalls = [num_examples * m["recall"] for num_examples, m in metrics]
    precisions = [num_examples * m["precision"] for num_examples, m in metrics]
    f1s = [num_examples * m["f1"] for num_examples, m in metrics]

    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "recall": sum(recalls) / sum(examples),
        "precision": sum(precisions) / sum(examples),
        "f1": sum(f1s) / sum(examples),
    }


def server_side_parameters():
    """Initialize model parameters on server side."""
    dataset_name = projectconf["config"]["dataset_name"]
    if dataset_name == "uoft-cs/cifar10":
        ndarrays = get_weights(Net())
    elif dataset_name == "ylecun/mnist":
        ndarrays = get_weights(Net(10, 1, 32, 1))

    parameters = ndarrays_to_parameters(ndarrays)

    return parameters


def get_evaluate_fn(model: torch.nn.Module, device: str = "cpu"):
    """Return an evaluation function for server-side evaluation using PyTorch (MNIST)."""
    dataset_name = projectconf["config"]["dataset_name"]

    if dataset_name == "uoft-cs/cifar10":
        # Transformaciones básicas para MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2535, 0.2616))  # Normalización CIFAR10
        ])
        # Dataset CIFAR-10
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    elif dataset_name == "ylecun/mnist":
        # Transformaciones básicas para MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Normalización MNIST
        ])
        # Dataset MNIST
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    # Usamos las últimas 5k imágenes como validación
    valloader = DataLoader(testset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # Actualizar pesos del modelo
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, dtype=model.state_dict()[k].dtype) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        model.to(device)
        model.eval()

        loss_total, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_total += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Guardar para métricas adicionales
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Métricas
        accuracy = correct / total
        avg_loss = loss_total / total
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        return avg_loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return evaluate


def create_configurations():
    """Returns complete configurations for strategies."""
    functions_dict = {
        "fit_weighted_average": fit_weighted_average,
        "evaluate_weighted_average": evaluate_weighted_average,
        "server_side": server_side_parameters,
        "evaluate_server": get_evaluate_fn,
        "None": None,
    }

    configurations = {
        "fraction_fit": projectconf["config"]["fraction_fit"],
        "fraction_evaluate": projectconf["config"]["fraction_evaluate"],
        "min_fit_clients": projectconf["config"]["min_fit_clients"],
        "min_evaluate_clients": projectconf["config"]["min_evaluate_clients"],
        "min_available_clients": projectconf["config"]["min_available_clients"],
        "evaluate_fn": functions_dict.get(projectconf["config"]["evaluate_fn"], None),
        "on_fit_config_fn": functions_dict.get(projectconf["config"]["on_fit_config_fn"], None),
        "on_evaluate_config_fn": functions_dict.get(projectconf["config"]["on_evaluate_config_fn"], None),
        "accept_failures": projectconf["config"]["accept_failures"],
        "initial_parameters": functions_dict.get(projectconf["config"]["initial_parameters"], None),
        "fit_metrics_aggregation_fn": functions_dict.get(projectconf["config"]["fit_metrics_aggregation_fn"], None),
        "evaluate_metrics_aggregation_fn": functions_dict.get(projectconf["config"]["evaluate_metrics_aggregation_fn"], None),
        "num_exec": num_exec,
        "strategy_name": strategy_name,
        "inplace": projectconf["config"]["inplace"]
    }

    # Map values from configuration to functions based on the dictionary
    for key, value in configurations.items():
        if value in functions_dict:
            configurations[key] = functions_dict[value]

    # Call function for initializing parameters
    if callable(configurations["initial_parameters"]):
        configurations["initial_parameters"] = configurations["initial_parameters"]()

    if configurations["evaluate_fn"] == get_evaluate_fn:
        dataset_name = projectconf["config"]["dataset_name"]
        if dataset_name == "uoft-cs/cifar10":
            model = Net()
        elif dataset_name == "ylecun/mnist":
            model = Net(10, 1, 32, 1)
        
        configurations["evaluate_fn"] = get_evaluate_fn(model, device="cpu")

    return configurations


def select_strategy():
    """Returns the selected strategy configurated."""
    configurations = create_configurations()  # Get configurations
    strategy_name = configurations["strategy_name"]

    strategy_mapping = get_strategy_mapping()
    selected_strategy_function = strategy_mapping.get(strategy_name, default_strategy)
    return selected_strategy_function(configurations)

def load_parameters():
    federation = projectconf["tempConfig"]["federation"]
    sub_execution = projectconf["tempConfig"]["execution_name"]
    
    checkpoint_path = projectconf["paths"]["checkpoint"].format(
        federation=federation, 
        strategy=strategy_name, 
        sub_execution=sub_execution, 
        num_exec=num_exec
    )

    pattern_file = "round-*-weights.npz"
    files = [file for file in os.listdir(checkpoint_path) if fnmatch.fnmatch(file, pattern_file)]
    
    if files:
        # Find the file with the highest round number
        latest_file = max(files, key=lambda file: int(re.search(r"round-(\d+)-weights\.npz", file).group(1)))

        weights = np.load(f"{checkpoint_path}/{latest_file}")
        parameters = fl.common.ndarrays_to_parameters([weights[key] for key in weights.files])

        return parameters
    return None

def server_fn(context: Context):
    """Configure and return a Flwr ServerAppComponents with CustomServer for semi-asynchrony."""
    num_rounds = projectconf["tempConfig"]["step_rounds"]
    offset = projectconf["tempConfig"]["last_round"]
    server_type = projectconf["config"]["server_type"]

    strategy = select_strategy()
    strategy.set_round_offset(offset)

    init_parameters = load_parameters()   
    if init_parameters is not None:
        strategy.initial_parameters = init_parameters

    config = ServerConfig(num_rounds=num_rounds)

    if server_type == 1:
        server = CustomServerSemiasync(client_manager=SimpleClientManager(), strategy=strategy)
    elif server_type == 2:
        server = CustomServerModSemiasync(client_manager=SimpleClientManager(), strategy=strategy)


    return ServerAppComponents(
        config=config,
        server=server,
    )


# Create ServerApp
app = ServerAppCustom(server_fn=server_fn)
