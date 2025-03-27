from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from app_code.task import (
    Net,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
    metrics_calculation,
)

import torch
import os
import time


class FlowerClient(NumPyClient):
    """FlowerClient handles training and evaluation of a PyTorch model on a local dataset."""

    def __init__(self, net, trainloader, valloader):
        """Init of the class FlowerClient."""
        self.net = net  # Model used for training
        self.trainloader = trainloader  # Training data
        self.valloader = valloader  # Validation data
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # Device used for training
        self.net.to(self.device)  # Send model to device

    def get_properties(self, config):
        """Returns properties of the client."""
        return {"user": os.getenv("DEVICE")}  # Return client device name

    def fit(self, parameters, config):
        """Train the model on the client's local dataset."""
        # Debug mode used for faking times on each client
        if config.get("debug", False):
            dev = os.getenv("DEVICE")
            if dev == "supernode-5":
                time.sleep(10)
            elif dev == "supernode-1":
                time.sleep(5)
            elif dev == "supernode-2":
                time.sleep(7)
            elif dev == "supernode-3":
                time.sleep(12)

            return (  # Debug mode returns emtpy metric values
                parameters,
                len(self.valloader.dataset),
                {
                    "accuracy": 0.0,
                    "loss": 0.0,
                    "loss_distributed": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
            )

        set_weights(self.net, parameters)

        train_loss, all_labels, all_preds = train(  # Train the model
            self.net, self.trainloader, self.device, config
        )

        if config.get(
            "evaluate_on_fit", False
        ):  # Evaluate the model if the flag is set, only for certain
            loss, _ = self.evaluate(parameters, {})  # strategies defined on the server

        else:  # If the flag is not set, return empty value of training loss = 0.0
            loss = 0.0

        # Calculate training metrics
        precision, recall, f1, accuracy = metrics_calculation(all_labels, all_preds)

        return (  # Returns real metric values
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "accuracy": accuracy.item(),
                "loss": loss,
                "loss_distributed": train_loss,
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's local dataset."""
        set_weights(
            self.net, parameters
        )  # Set model weights to the ones received from the server (aggregated weights)
        loss, accuracy, all_labels, all_preds = test(
            self.net, self.valloader, self.device
        )  # Evaluate the model

        precision, recall, f1, accuracy = metrics_calculation(
            all_labels, all_preds
        )  # Calculate evaluation metrics

        return (  # Returns calculated metric values
            loss,
            len(self.valloader.dataset),
            {
                "accuracy": accuracy.item(),
                "precision": precision.item(),
                "recall": recall.item(),
                "f1": f1.item(),
            },
        )


def client_fn(context: Context):
    """Load data and return a Flower client."""
    net = Net()  # Create model instance
    partition_id = context.node_config["partition-id"]  # Get partition id from context
    num_partitions = context.node_config[
        "num-partitions"
    ]  # Get total number of partitions from context
    partition_type = context.node_config["partition-type"]
    trainloader, valloader = load_data(
        partition_id, num_partitions, partition_type
    )  # Create train and val loaders

    # Return Client instance
    return FlowerClient(net, trainloader, valloader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
