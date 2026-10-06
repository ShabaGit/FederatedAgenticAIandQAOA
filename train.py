import torchquantum as tq
import torchquantum.functional as tqf
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
import numpy as np

NUM_CLIENTS = 30
NUM_GLOBAL_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.002
NUM_QUBITS = 4
DIRICHLET_ALPHA = 0.3

CLASSES = ["AnnualCrop", "Forest"]

class BinaryEuroSATDataset(Dataset):
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names
        self.class_indices = [dataset.class_to_idx[class_name] for class_name in class_names]
        self.samples = []

        for idx, label in enumerate(dataset.targets):
            if label in self.class_indices:
                new_label = self.class_indices.index(label)
                self.samples.append((idx, new_label))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        original_idx, new_label = self.samples[idx]
        image, _ = self.dataset[original_idx]
        return image, new_label

class HybridQuantumModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, op_list=None):
            super().__init__()
            self.n_wires = NUM_QUBITS
            self.random_layer = tq.RandomLayer(n_ops=5, wires=list(range(self.n_wires)))
            if op_list is not None:
                self.random_layer.op_list = op_list
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.h(qdev, wires=3)
            tqf.sx(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])

    def __init__(self, op_list=None):
        super().__init__()
        self.n_wires = NUM_QUBITS
        self.classical_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(256, 16)
        )
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])
        self.q_layer = self.QLayer(op_list=op_list)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(NUM_QUBITS, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.classical_cnn(x)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch_size, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        x = self.measure(qdev)
        x = x.reshape(batch_size, NUM_QUBITS)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def load_eurosat():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_dir = "./EuroSAT"
    os.makedirs(data_dir, exist_ok=True)

    dataset = datasets.EuroSAT(
        root=data_dir,
        download=True,
        transform=transform
    )

    dataset = BinaryEuroSATDataset(dataset, CLASSES)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    return train_dataset, test_dataset

def get_labels(dataset):
    labels = []

    for i in range(len(dataset)):
        if isinstance(dataset, Subset):
            original_index = dataset.indices[i]
            label = dataset.dataset.targets[original_index]
        else:
            label = dataset.targets[i]

        labels.append(label)

    return np.array(labels)

def split_dirichlet(dataset, num_clients, alpha=0.1):
    labels = get_labels(dataset)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(
            np.repeat(alpha, num_clients)
        )

        proportions = proportions / proportions.sum()

        split_points = (
            np.cumsum(proportions) * len(class_indices)
        ).astype(int)[:-1]

        class_splits = np.split(
            class_indices,
            split_points
        )

        for client_id in range(num_clients):
            client_indices[client_id].extend(
                class_splits[client_id].tolist()
            )

    client_datasets = []

    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

        client_dataset = Subset(
            dataset,
            client_indices[client_id]
        )

        client_datasets.append(client_dataset)

    return client_datasets

def print_client_distribution(client_datasets):
    print("\nDirichlet Non-IID Client Distribution")
    print(f"Alpha = {DIRICHLET_ALPHA}\n")

    for client_id, dataset in enumerate(client_datasets):
        labels = []

        for index in dataset.indices:
            if isinstance(dataset.dataset, Subset):
                original_index = dataset.dataset.indices[index]
                label = dataset.dataset.dataset.targets[original_index]
            else:
                label = dataset.dataset.targets[index]

            labels.append(label)

        unique, counts = np.unique(
            labels,
            return_counts=True
        )

        distribution = dict(
            zip(unique, counts)
        )

        print(f"Client {client_id + 1}: {len(dataset)} samples")
        print(f"Class distribution: {distribution}\n")

def train_client(global_model, client_dataset, device):
    local_model = copy.deepcopy(global_model)
    local_model.train()

    optimizer = optim.Adam(
        local_model.parameters(),
        lr=LEARNING_RATE
    )

    train_loader = DataLoader(
        client_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    for local_epoch in range(LOCAL_EPOCHS):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = local_model(images)

            loss = F.nll_loss(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

    return copy.deepcopy(local_model.state_dict()), len(client_dataset)

def fedavg(client_weights, client_sizes):
    total_samples = sum(client_sizes)
    global_weights = copy.deepcopy(client_weights[0])

    for key in global_weights.keys():
        if torch.is_floating_point(global_weights[key]):
            global_weights[key] = torch.zeros_like(global_weights[key])

            for client_id in range(len(client_weights)):
                weight = client_sizes[client_id] / total_samples

                global_weights[key] += (
                    client_weights[client_id][key] * weight
                )

        else:
            global_weights[key] = client_weights[0][key]

    return global_weights

def evaluate(model, test_dataset, device):
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = F.nll_loss(
                outputs,
                labels,
                reduction="sum"
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    test_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return test_loss, accuracy

def federated_training(global_model, client_datasets, test_dataset, device):
    for round_id in range(NUM_GLOBAL_ROUNDS):
        client_weights = []
        client_sizes = []

        for client_id, client_dataset in enumerate(client_datasets):
            weights, data_size = train_client(
                global_model,
                client_dataset,
                device
            )

            client_weights.append(weights)
            client_sizes.append(data_size)

        global_weights = fedavg(
            client_weights,
            client_sizes
        )

        global_model.load_state_dict(
            global_weights
        )

        test_loss, accuracy = evaluate(
            global_model,
            test_dataset,
            device
        )

        print(
            f"Global Round {round_id + 1}/{NUM_GLOBAL_ROUNDS} | "
            f"Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}%"
        )

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Binary classes: {CLASSES}")

    train_dataset, test_dataset = load_eurosat()

    client_datasets = split_dirichlet(
        train_dataset,
        NUM_CLIENTS,
        alpha=DIRICHLET_ALPHA
    )

    print_client_distribution(
        client_datasets
    )

    random_layer = tq.RandomLayer(
        n_ops=5,
        wires=list(range(NUM_QUBITS))
    )

    op_list = random_layer.op_list

    global_model = HybridQuantumModel(
        op_list=op_list
    ).to(device)

    federated_training(
        global_model,
        client_datasets,
        test_dataset,
        device
    )

if __name__ == "__main__":
    main()