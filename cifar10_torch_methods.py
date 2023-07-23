"""
Mostly taken from RayTune docs: 
https://docs.ray.io/en/latest/tune/examples/includes/pbt_tune_cifar10_with_keras.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import random

from scipy.stats import loguniform
from scipy.stats import uniform

# NUM_SAMPLES = 128 #use this for debuging (small training data) FAST
NUM_SAMPLES = None  # use this in real situations (all trainng data) SLOW
num_classes = 10


class ConfigGeneratorCifar10:
    def __init__(self, random_state=None):
        if random_state == None:
            random_state = random.randint(0, 9999)
        random.seed(random_state)
        np.random.seed(random_state)

        self.n_sampled = 0

    def get_hyperparameter_configuration(self, n):
        """
		returns n configurations
		"""
        T = []
        for _ in range(n):
            config = {
                "num_layers": np.random.choice([2, 3, 4]),
                "num_filters": np.random.choice([16, 32, 48, 64]),
                "batch_size": np.random.choice([64, 128, 256, 512]),
                "learning_rate": loguniform.rvs(1e-4, 1e-1),
                "momentum": loguniform.rvs(10e-5, 0.9),
            }
            id = str(self.n_sampled)
            t = {"config": config, "id": id, "curve": []}
            T.append(t)
            self.n_sampled = self.n_sampled + 1
        return T


def train_cifar(config: dict, id: str, epochs: int, dir_name: str):

    torch.manual_seed(1)

    # create model
    model = build_model(config)

    model_file = "./" + dir_name + "/cifar_" + id + ".pt"

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model.load_state_dict(torch.load(f))

    # move to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("learning_rate"),
        momentum=config.get("momentum"),
    )

    # load data
    trainset, testset = read_data()

    # define dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=int(config.get("batch_size")), shuffle=True, num_workers=0
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=int(config.get("batch_size")), shuffle=False, num_workers=0
    )

    val_acc = []
    # training loop
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # test loss
        # test_loss = 0.0
        # test_steps = 0
        total = 0
        correct = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # test_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc.append((correct / total) * -1)

    with open(model_file, "wb") as f:
        torch.save((model.state_dict()), f)

    return val_acc


def read_data():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="/p/project/cslfse/aach1/NAS/cifar/data/",
        train=True,
        download=False,
        transform=transform_train,
    )

    testset = torchvision.datasets.CIFAR10(
        root="/p/project/cslfse/aach1/NAS/cifar/data/",
        train=False,
        download=False,
        transform=transform_test,
    )

    return trainset, testset


def build_model(config: dict):

    num_layers = config.get("num_layers")
    num_filters = config.get("num_filters")

    # build the model
    model = nn.Sequential(nn.Conv2d(3, num_filters, 5), nn.MaxPool2d(2, 2), nn.ReLU())

    for i in range(num_layers - 2):
        model.add_module(
            "conv_middle_{}".format(i), nn.Conv2d(num_filters, num_filters, 5)
        )
        model.add_module("relu_middle_{}".format(i), nn.ReLU())

    model.add_module("conv_last", nn.Conv2d(num_filters, num_filters * 2, 5))
    model.add_module("relu_conv_last", nn.ReLU())

    model.add_module("flatten", nn.Flatten())

    model.add_module(
        "fc_1",
        nn.Linear(
            int(num_filters * 2 * (((32 - 4) / 2 - (4 * (num_layers - 1)))) ** 2), 120
        ),
    )

    model.add_module("relu_fc_1", nn.ReLU())
    model.add_module("fc_2", nn.Linear(120, 84))
    model.add_module("relu_fc_2", nn.ReLU())
    model.add_module("fc_3", nn.Linear(84, 10))

    return model
