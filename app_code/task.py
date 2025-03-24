from collections import OrderedDict
from nets.mobilenet import MobileNetV1

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner


class Net(MobileNetV1):
    """Custom MobileNetV1"""
    def __init__(self, num_classes=10):
        super().__init__(num_classes)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int, partition_type: str):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if partition_type == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)

        elif partition_type == "noniid":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=0.5,
                self_balancing=True,
                shuffle=False,
                seed=42
                )
            
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(       
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def metrics_calculation(y_true, y_pred, num_classes=10):
    precision_values = []
    recall_values = []
    correct_preds = 0  # Contador de predicciones correctas
    total_preds = len(y_true)  # Total de muestras (tamaño de y_true)

    # Calculamos precisión y recall para cada clase
    for class_label in range(num_classes):
        true_positives = ((y_true == class_label) & (y_pred == class_label)).sum().float()
        false_positives = ((y_true != class_label) & (y_pred == class_label)).sum().float()
        false_negatives = ((y_true == class_label) & (y_pred != class_label)).sum().float()

        # Precisión y recall para la clase actual
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        precision_values.append(precision)
        recall_values.append(recall)
        
        # Contamos los verdaderos positivos (TP) de cada clase para calcular la accuracy
        correct_preds += true_positives

    # Promedio de precisión y recall sobre todas las clases
    precision_avg = torch.mean(torch.tensor(precision_values))
    recall_avg = torch.mean(torch.tensor(recall_values))

    # Cálculo del F1 score
    f1 = 2 * (precision_avg * recall_avg) / (precision_avg + recall_avg + 1e-8)

    # Calcular la exactitud (accuracy)
    accuracy = correct_preds / total_preds

    return precision_avg, recall_avg, f1, accuracy


def slice_data(dataloader, config):
    subset_size = config.get("subset_size", 1024)
    inner_round = config.get("inner_round", 1)

    print(inner_round)

    start = (inner_round - 1) * subset_size
    end = inner_round * subset_size

    start = start % len(dataloader.dataset)
    end = end % len(dataloader.dataset)

    if start < end:
        train_indices = list(range(start, end))
    else:
        train_indices = list(range(start, len(dataloader.dataset))) + list(range(0, end))
    
    train_subset = Subset(dataloader.dataset, train_indices)
    sliced_dataloader = DataLoader(train_subset, batch_size=dataloader.batch_size, shuffle=True)

    return sliced_dataloader


def train(net, trainloader, device, config):
    """Train the model on the training set."""
    epochs = config.get("epochs", 1)
    proximal_mu = config.get('proximal_mu', 0.0)

    sliced_trainloader = slice_data(trainloader, config)

    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    net.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    global_params = [param.clone().detach() for param in net.parameters()]

    for _ in range(epochs):
        for batch in sliced_trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            outputs = net(images.to(device))

            proximal_term = 0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)

            loss = criterion(outputs, labels.to(device)) + (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    avg_trainloss = running_loss / len(sliced_trainloader)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    return avg_trainloss, all_labels, all_preds


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            _, preds = torch.max(outputs, 1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    return loss, accuracy, all_labels, all_preds


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
    net.load_state_dict(state_dict, strict=True)
