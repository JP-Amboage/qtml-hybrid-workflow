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


class ConfigGeneratorTinyImg:
    def __init__(self, random_state=None):
        if random_state == None:
            random_state = random.randint(0, 9999)
        random.seed(random_state)
        np.random.seed(random_state)
        # tf.random.set_seed(random_state)
        # tf.keras.utils.set_random_seed(random_state)
        # os.environ['PYTHONHASHSEED'] = str(random_state)
        # os.environ['TF_DETERMINISTIC_OPS'] = '1'

        self.n_sampled = 0

    def get_hyperparameter_configuration(self, n):
        """
		returns n configurations
		"""
        T = []
        for _ in range(n):
            config = {
                "batch_size": np.random.choice([64, 128, 256, 512]),
                "learning_rate": loguniform.rvs(1e-4, 1e-1),
                "momentum": loguniform.rvs(10e-5, 0.9),
            }
            id = str(self.n_sampled)
            t = {"config": config, "id": id, "curve": []}
            T.append(t)
            self.n_sampled = self.n_sampled + 1
        return T


def train_tinyimg(config: dict, id: str, epochs: int, dir_name: str):

    torch.manual_seed(1)

    # create model
    model = build_model(config)

    model_file = "./" + dir_name + "/tiny_img_" + id + ".pt"

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
            transforms.RandomCrop(56),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(0.3),
            transforms.ToTensor(),  # Converting cropped images to tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data_dir = "/p/project/prcoe12/tiny-imagenet-200/train"
    test_data_dir = "/p/project/prcoe12/tiny-imagenet-200/val"

    trainset = torchvision.datasets.ImageFolder(
        train_data_dir, transform=transform_train
    )

    testset = torchvision.datasets.ImageFolder(test_data_dir, transform=transform_test)

    return trainset, testset


def build_model(config: dict):

    model = torchvision.models.resnet18()

    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.maxpool = nn.Sequential()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 200)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = 200

    return model
