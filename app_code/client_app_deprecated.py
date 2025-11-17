import os

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app_code.task import (
    Net,
    get_weights,
    load_data,
    set_weights,
    test,
    train,
)


class FlowerClient(NumPyClient):
    """FlowerClient handles training and evaluation of a PyTorch model on a local dataset."""

    def __init__(self, net, trainloader, newloader, valloader, partition_id):
        """Init of the class FlowerClient."""
        self.net = net  # Model used for training
        self.trainloader = trainloader  # Training data
        self.newloader = newloader  # New incoming data (not used yet, usually empty)
        self.valloader = valloader  # Validation data
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Device used for training
        self.net.to(self.device)  # Send model to device
        self.partition_id = partition_id

    def get_properties(self, config):
        """Returns properties of the client."""
        user = os.getenv("DEVICE")  # Get device username

        # Properties used to calculate client utility
        labels = self.trainloader.tensors[1]    # List of all labels on data
        unique_classes = torch.unique(labels).tolist()  # List of unique labels
        quantity_data = len(self.trainloader)   # Integer with amount of data
        
        return {"user":             user,   # Return username
                "labels":           ",".join(map(str, unique_classes)), # Return string separated by commas of unique labels of client
                "quantity_data":    quantity_data} # Return data quantity

    def fit(self, parameters, config):
        """Train the model on the client's local dataset."""
        set_weights(self.net, parameters)

        """if config.get("server_round") == 1:
            offset = 100
            pace = 10
            timeout = 200
            init_sliced_windows(self.trainloader, config)

            producer_thread = threading.Thread(target=producer, args=(self.newloader, offset, pace, config.get("batch_size", 32)), daemon=True)
            consumer_thread = threading.Thread(target=consumer, args=(config, timeout,), daemon=True)

            producer_thread.start()
            consumer_thread.start()"""

        loss, y_true, y_pred = train(  # Train the model
            self.net, self.device, config, self.trainloader,
        )

        # Calculate training metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return (  # Returns calculated metric values
            get_weights(self.net),
            len(self.trainloader),
            {
                "accuracy": accuracy,
                "loss": loss,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's local dataset."""      
        set_weights(
            self.net, parameters
        )  # Set model weights to the ones received from the server (aggregated weights)

        loss, accuracy, y_true, y_pred = test(
            self.net, self.valloader, self.device
        )  # Evaluate the model

        # Calculate training metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        return (  # Returns calculated metric values
            loss,
            len(self.valloader.dataset),
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
        )


def client_fn(context: Context):
    """Load data and return a Flower client."""
    net = Net()  # Create model instance
    partition_id = context.node_config["partition-id"]  # Get partition id from context
    num_partitions = context.node_config["num-partitions"]  # Get total number of partitions from context
    partition_type = context.node_config["partition-type"]
    trainloader, newloader, valloader = load_data(partition_id, num_partitions, partition_type)
    # Return Client instance
    return FlowerClient(net, trainloader, newloader, valloader, partition_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
