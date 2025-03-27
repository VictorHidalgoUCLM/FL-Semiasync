from collections import OrderedDict
from app_code.nets.mobilenet import MobileNetV1

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner


class Net(MobileNetV1):
    """Custom MobileNetV1 class. Defaults to 10 classes."""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, partition_type: str):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:  # If dataset has not been loaded before
        # Load data with an iid partition
        if partition_type == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)

        # Load data with a non-iid partition
        elif partition_type == "noniid":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=0.5,  # Non-iidness
                self_balancing=True,
                shuffle=False,
                seed=42,
            )  # Unbalances number of classes on each partition

        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )  # Update FederatedDataset

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )  # Normalization of data in partition

    def apply_transforms(batch):
        """Apply transforms (normalization) to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def metrics_calculation(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 10
) -> tuple:
    """Calculates performance metrics for a multi-class classification problem."""
    precision_values = []
    recall_values = []
    correct_preds = 0
    total_preds = len(y_true)  # Total de muestras (tamaño de y_true)

    # Calculates precision and recall for each class label
    for class_label in range(num_classes):
        true_positives = (
            ((y_true == class_label) & (y_pred == class_label)).sum().float()
        )  # Gets true positives (TP)
        false_positives = (
            ((y_true != class_label) & (y_pred == class_label)).sum().float()
        )  # Gets false positives (FP)
        false_negatives = (
            ((y_true == class_label) & (y_pred != class_label)).sum().float()
        )  # Gets false negatives (FN)

        # Precisión and recall for current class label
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        precision_values.append(precision)
        recall_values.append(recall)

        # True Positives (TP) of each class for accuracy calculation
        correct_preds += true_positives

    # Mean for precision and recall
    precision_avg = torch.mean(torch.tensor(precision_values))
    recall_avg = torch.mean(torch.tensor(recall_values))

    # F1 score and accuracy
    f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg + 1e-8)
    accuracy = correct_preds / total_preds

    return precision_avg, recall_avg, f1, accuracy


def slice_data(dataloader: DataLoader, config: dict) -> DataLoader:
    """Slice the data from the dataloader based on the configuration."""
    # Get config values
    subset_size = config.get("subset_size", 1024)
    inner_round = config.get("inner_round", 1)

    # Implementation of the sliding window for data selection:
    # Calculate start and end indices for slicing
    start = (inner_round - 1) * subset_size
    end = inner_round * subset_size

    # Ensure indices wrap around the dataset size
    start = start % len(dataloader.dataset)
    end = end % len(dataloader.dataset)

    # Handle slicing logic for wrapping around the dataset
    if start < end:
        train_indices = list(range(start, end))
    else:
        train_indices = list(range(start, len(dataloader.dataset))) + list(
            range(0, end)
        )

    # Create a subset of the dataset and a new DataLoader
    train_subset = Subset(dataloader.dataset, train_indices)
    sliced_dataloader = DataLoader(
        train_subset, batch_size=dataloader.batch_size, shuffle=True
    )

    return sliced_dataloader


def train(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
) -> tuple:
    """Train the model on the training set."""
    # Get configuration values for training
    epochs = config.get("epochs", 1)  # Number of epochs to train. Default is 1.
    proximal_mu = config.get(
        "proximal_mu", 0.0
    )  # Proximal term coefficient for FedProx. Default is 0.0.

    sliced_trainloader = slice_data(trainloader, config)

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)  # Loss function
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.1
    )  # Optimizer with learning rate 0.1

    net.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    # Store a copy of the global model parameters for FedProx
    global_params = [param.clone().detach() for param in net.parameters()]

    # Training loop for the specified number of epochs
    for _ in range(epochs):
        for batch in sliced_trainloader:
            images = batch["img"]  # Extract images from the batch
            labels = batch["label"]  # Extract labels from the batch
            optimizer.zero_grad()  # Reset gradients
            outputs = net(images.to(device))  # Forward pass

            # Calculate the proximal term for FedProx
            # Note that if proximal_mu = 0, it is the same as FedAvg
            proximal_term = 0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)

            # Compute the total loss (cross-entropy + proximal term)
            loss = (
                criterion(outputs, labels.to(device))
                + (proximal_mu / 2) * proximal_term
            )

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            running_loss += loss.item()  # Accumulate loss
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            all_labels.append(labels.cpu())  # Store true labels
            all_preds.append(preds.cpu())  # Store predicted labels

    avg_trainloss = running_loss / len(sliced_trainloader)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    return avg_trainloss, all_labels, all_preds


def test(
    net: torch.nn.Module, testloader: torch.utils.data.DataLoader, device: torch.device
) -> tuple:
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # Loss function
    correct, loss = 0, 0.0
    all_labels = []
    all_preds = []

    # Disable gradient computation for testing
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)  # Move images to the specified device
            labels = batch["label"].to(device)  # Move labels to the specified device
            outputs = net(images)  # Forward pass
            loss += criterion(outputs, labels).item()  # Accumulate loss
            correct += (
                (torch.max(outputs.data, 1)[1] == labels).sum().item()
            )  # Count correct predictions
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    # Calculate accuracy and average loss
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    # Concatenate all true and predicted labels into tensors
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    return loss, accuracy, all_labels, all_preds


def get_weights(net):
    """Returns weights of the local net."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Sets weights with given parameters to the local net."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    net.load_state_dict(state_dict, strict=True)
