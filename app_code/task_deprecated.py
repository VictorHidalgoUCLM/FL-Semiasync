import queue
import random
import time
from collections import OrderedDict
from queue import Empty
import numpy as np

import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor

from app_code.nets.mobilenet import MobileNetV1_MNIST, MobileNetV3Small_CIFAR10


class Net(MobileNetV1_MNIST):
    """Custom MobileNetV1 class. Defaults to 10 classes."""

    def __init__(self, num_classes=10):
        super().__init__(num_classes)


fds = None  # Cache FederatedDataset
sliced_windows = []
incoming_data = []
data_queue = queue.Queue()
seed = 42

def end_queue():
    global data_queue
    data_queue.put(None)

def producer(dataloader: DataLoader, offset=500, delay=20, batch_size=32):
    global data_queue
   
    time.sleep(offset)

    for batch in dataloader:
        if batch["label"].size(0) == batch_size:
            data_queue.put(batch)
        else:
            pass

        time.sleep(delay)

def consumer(config: dict, timeout: float = 600.0):
    global data_queue
    global incoming_data
    """global timeout_global"""

    subset_size = config.get("subset_size", 1024)
    """timeout_global = timeout"""

    while True:
        try:
            batch = data_queue.get(timeout=600.0)
            if batch is None:
                print("Ended receiving data...")
                """timeout_global = True"""
                break

            batch_size = batch["label"].size(0)
            batch_dataset = TensorDataset(batch["img"], batch["label"])

            for i in range(len(incoming_data)):
                incoming_window = incoming_data[i]
                indices_sum = 0

                for j in range(len(incoming_window)):
                    dataset = incoming_window[j]
                    indices_sum += len(dataset)

                if indices_sum >= subset_size:
                    continue
                
                else:
                    new_window_length = indices_sum + batch_size

                    if new_window_length <= subset_size:
                        incoming_window.append(batch_dataset)

                        if new_window_length == subset_size:
                            incoming_data.append([])

                    else:
                        remaining_space = subset_size - indices_sum
                        
                        batch_1 = {
                            'img': batch['img'][:remaining_space],
                            'label': batch['label'][:remaining_space]
                        }
                        batch_2 = {
                            'img': batch['img'][remaining_space:],
                            'label': batch['label'][remaining_space:]
                        }

                        batch_1_dataset = TensorDataset(batch_1["img"], batch_1["label"])
                        incoming_window.append(batch_1_dataset)

                        batch_2_dataset = TensorDataset(batch_2["img"], batch_2["label"])
                        incoming_data.append([batch_2_dataset])

            data_queue.task_done()

        except Empty:
            print("Timeout... Ending receiving data...")
            break
        
    end_queue()

def update_sliced_windows(trainloader, config: dict):
    global incoming_data

    subset_size = config.get("subset_size", 1024)

    if (
        len(incoming_data) > 0
        and len(incoming_data[0]) > 0
        and len(ConcatDataset(incoming_data[0])) >= subset_size
    ):
        trainloader = ConcatDataset([trainloader, ConcatDataset(incoming_data[0])])
        incoming_data.pop(0)

        return True
    else:
        return False
            
def load_data(partition_id: int, num_partitions: int, partition_type: str):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds

    if fds is None:  # If dataset has not been loaded before
        # Load data with an iid partition
        if partition_type == "iid":
            train_partitioner = IidPartitioner(num_partitions=num_partitions)

        # Load data with a non-iid partition
        elif partition_type == "noniid":
            train_partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=0.5,  # Non-iidness
                self_balancing=True,
                shuffle=False
            )  # Unbalances number of classes on each partition

        test_partitioner = IidPartitioner(num_partitions=num_partitions)    # Test partitioner

        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": train_partitioner,
                          "test": test_partitioner},
        )  # Update FederatedDataset

    # Get specific partition for the client
    partition_train = fds.load_partition(partition_id, "train")
    partition_test = fds.load_partition(partition_id, "test")

    """pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465),
                  (0.2470, 0.2535, 0.2616))]
    )  # Normalization of data in partition"""

    pytorch_transforms = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))  # MNIST: 1 canal
    ])


    def apply_transforms(batch):
        """Apply transforms (normalization) to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    # Apply transforms
    partition_train = partition_train.with_transform(apply_transforms)
    partition_test = partition_test.with_transform(apply_transforms)

    classes_list = list(range(len(partition_train.features['label'].names)))
    #classes_list = list(range(10))
    
    def filter_by_label(dataset, labels1, labels2, ratio1=1.0, ratio2=1.0, seed=42):
        if not isinstance(labels1, list):
            labels1 = [labels1]
        if not isinstance(labels2, list):
            labels2 = [labels2]

        random.seed(seed)  # Para reproducibilidad

        # Identificar etiquetas comunes
        common_labels = set(labels1) & set(labels2)
        only_labels1 = set(labels1) - common_labels

        common_labels_idxs = []
        only_labels1_idxs = []

        # Clasificamos los índices
        for i, data in enumerate(dataset):
            label = data["label"]
            if label in only_labels1:
                only_labels1_idxs.append(i)
            elif label in common_labels:
                common_labels_idxs.append(i)

        # Barajamos y dividimos los índices con etiquetas comunes
        random.shuffle(only_labels1_idxs)
        split1 = int(len(only_labels1_idxs) * ratio1)
        subset1_idxs = only_labels1_idxs[:split1]
        subset2_idxs = only_labels1_idxs[split1:]

        random.shuffle(common_labels_idxs)
        split2 = int(len(common_labels_idxs) * ratio2)
        subset1_idxs += common_labels_idxs[:split2]
        subset2_idxs += common_labels_idxs[split2:]

        # Creamos tensores para subset1
        imgs1 = torch.cat([dataset[i]["image"].unsqueeze(0) for i in subset1_idxs], dim=0)
        labels1_tensor = torch.tensor([dataset[i]["label"] for i in subset1_idxs])

        # Creamos el nuevo dataset reducido
        reduced_dataset = TensorDataset(imgs1, labels1_tensor)

        return reduced_dataset, Subset(dataset, subset2_idxs)

    def get_partition_classes(classes_list, partition_id, num_partitions):
        """Get classes for a specific partition."""
        classes_per_partition = len(classes_list) // num_partitions
        start_idx = partition_id * classes_per_partition
        end_idx = start_idx + classes_per_partition
        return classes_list[start_idx:end_idx]

    classes_subset = get_partition_classes(classes_list, partition_id, num_partitions)
    original_subset, new_subset = filter_by_label(partition_train, classes_list, classes_subset)
    trainloader = original_subset

    if len(new_subset) <= 0:
        newloader = DataLoader(TensorDataset(torch.Tensor([])), batch_size=32, shuffle=False)
    else:
        newloader = DataLoader(new_subset, batch_size=32, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=32)

    return trainloader, newloader, testloader

def select_window(config: dict) -> DataLoader:
    """Slice the data from the dataloader based on the configuration."""
    global sliced_windows

    # Get config values
    inner_round = config.get("inner_round", 1)

    # Calculate needed window index
    index = (inner_round - 1) % len(sliced_windows)

    print(f"Selected windows index: {index}")

    # Extract data from sliced_windows
    window = sliced_windows[index]
    dataset = ConcatDataset(window)

    # Create a subset of the dataset and a new DataLoader
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    return dataloader

def init_sliced_windows(dataset, config: dict):
    global sliced_windows
    global incoming_data
    global seed
    global reduced_dataset

    g = torch.Generator()
    g.manual_seed(seed)
    seed += 1

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)

    subset_size = config.get("subset_size", 1024)
    total_windows = len(dataset) // subset_size
    total_data = total_windows * subset_size
    inner_sum = 0
    i = 0

    sliced_windows = [[] for _ in range(total_windows)]
    incoming_data.append([])

    for batch in dataloader:
        images, labels = batch
        batch_size = labels.size(0)

        if i >= total_windows:
            # Ya llenamos todas las ventanas -> datos sobrantes van al resto
            incoming_data[0].append(TensorDataset(images, labels))
            continue

        if inner_sum + batch_size <= subset_size:
            sliced_windows[i].append(TensorDataset(images, labels))

            if inner_sum + batch_size == subset_size:
                i += 1
                inner_sum = 0

            else:
                inner_sum += batch_size

        else:
            # Dividir el batch: parte va a la ventana actual, el resto a la siguiente
            remaining = subset_size - inner_sum
            sliced_windows[i].append(
                TensorDataset(images[:remaining], labels[:remaining])
            )
            i += 1
            inner_sum = 0

            if i < total_windows:
                sliced_windows[i].append(
                    TensorDataset(images[remaining:], labels[remaining:])
                )
                inner_sum = labels[remaining:].size(0)
            else:
                incoming_data[0].append(
                    TensorDataset(images[remaining:], labels[remaining:])
                )

    reduced_dataset = Subset(dataset, list(range(total_data)))

def shuffle_sliced_windows(dataset, config: dict):
    global sliced_windows
    global seed

    g = torch.Generator()
    g.manual_seed(seed)
    seed += 1

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)

    subset_size = config.get("subset_size", 1024)
    total_windows = len(dataset) // subset_size
    inner_sum = 0
    i = 0

    sliced_windows = [[] for _ in range(total_windows)]

    for batch in dataloader:
        images, labels = batch
        batch_size = labels.size(0)

        if inner_sum + batch_size <= subset_size:
            sliced_windows[i].append(TensorDataset(images, labels))

            if inner_sum + batch_size == subset_size:
                i += 1
                inner_sum = 0

            else:
                inner_sum += batch_size

        else:
            sliced_windows[i].append(TensorDataset(images, labels))
            i += 1
            inner_sum += batch_size

def train(
    net: torch.nn.Module,
    device: torch.device,
    config: dict,
    trainloader: DataLoader,
) -> tuple:
    global reduced_dataset
    #n = 5

    """Train the model on the training set."""
    # Get configuration values for training
    epochs = config.get("epochs", 1)  # Number of epochs to train. Default is 1.
    proximal_mu = config.get(
        "proximal_mu", 0.0
    )  # Proximal term coefficient for FedProx. Default is 0.0.

    """server_round = config.get("server_round", 0)
    print(f"Round: {server_round}")

    updated = update_sliced_windows(trainloader, config)
    if server_round % n == n-1 or updated:
        print("Reshuffling data...")
        shuffle_sliced_windows(reduced_dataset, config)"""

    """if updated:
        opportunity += 1
        print("Updated!")

    if early_stopping.early_stop:
        print("Early stopping detected :(")

        if not updated:
            opportunity -= 1
    
        net.load_state_dict(early_stopping.checkpoint)

        if opportunity > 0:
            shuffle_sliced_windows(reduced_dataset, config)   
            early_stopping.restart()
        else:
            raise ValueError("Client has no opportunities left...")

    elif server_round % n == n-1 or updated:
        print("Reshuffling data...")
        shuffle_sliced_windows(reduced_dataset, config)

    #train_window = select_window(config)    # Get DataLoader sliced"""

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)  # Loss function
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.01
    )  # Optimizer with learning rate 0.01

    net.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []

    # Store a copy of the global model parameters for FedProx
    global_params = [param.clone().detach() for param in net.parameters()]
    dataloader = DataLoader(
        trainloader, batch_size=32, shuffle=False
    )

    # Training loop for the specified number of epochs
    for _ in range(epochs):
        #for batch in train_window:
        for batch in dataloader:
            #images = batch["img"]  # Extract images from the batch
            #label = batch["label"]  # Extract labels from the batch
            
            images, label = batch
            optimizer.zero_grad()  # Reset gradients
            outputs = net(images.to(device))  # Forward pass

            # Calculate the proximal term for FedProx
            # Note that if proximal_mu = 0, it is the same as FedAvg
            proximal_term = 0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)

            # Compute the total loss (cross-entropy + proximal term)
            loss = (
                criterion(outputs, label.to(device))
                + (proximal_mu / 2) * proximal_term
            )

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            running_loss += loss.item()  # Accumulate loss
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            all_labels.append(label.cpu())  # Store true labels
            all_preds.append(preds.cpu())  # Store predicted labels

    avg_trainloss = running_loss / len(trainloader)
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
    