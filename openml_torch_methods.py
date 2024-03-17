import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import os
import random

from scipy.stats import loguniform
from scipy.stats import uniform

from sklearn.model_selection import train_test_split

import pandas as pd

import openml

os.mkdir('/tmp/openml') 


# NUM_SAMPLES = 128 #use this for debuging (small training data) FAST
NUM_SAMPLES = None  # use this in real situations (all trainng data) SLOW
num_classes = 10


class ConfigGeneratorOpenML:
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
                "alpha": loguniform.rvs(1e-8, 1),
                "batch_size": np.random.choice([64, 128, 256]),
                "depth": np.random.choice([1, 2, 3]),
                "learning_rate": loguniform.rvs(1e-5, 1),
                "width": np.random.choice([16, 32, 64, 128, 256, 512, 1024]),
            }
            id = str(self.n_sampled)
            t = {"config": config, "id": id, "curve": []}
            T.append(t)
            self.n_sampled = self.n_sampled + 1
        return T


def train_openml(config: dict, id: str, epochs: int, dir_name: str):
    
    openml.config.set_root_cache_directory('/tmp/openml')

    torch.manual_seed(1)
    
    dataset_id = os.getenv("OPENML_ID")
    
    # Load OpenML dataset
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=[name for name, is_categorical in zip(attribute_names, categorical_indicator) if is_categorical], dtype=int)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=int(config.get("batch_size")), shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(config.get("batch_size")), shuffle=False)

    # Initialize the network
    input_size = X_train.shape[1]
    output_size = 1  # Assuming regression problem

    # create model
    model = build_model(input_size, output_size, config)

    model_file = "./" + dir_name + "/openml_" + str(dataset_id) + "_" + id + ".pt"

    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model.load_state_dict(torch.load(f))

    # move to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # define optimizer and loss
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate"),
        weight_decay=config.get("alpha"),
    )

    val_losses = []
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_outputs = model(X_batch)
                val_loss = criterion(val_outputs, y_batch)
                val_loss_sum += val_loss.item()
        avg_val_loss = val_loss_sum / len(test_loader)
        val_losses.append(avg_val_loss)

    with open(model_file, "wb") as f:
        torch.save((model.state_dict()), f)

    return val_losses


def build_model(input_size, output_size, config: dict):

    class MLPNetwork(nn.Module):
        def __init__(self, input_size, output_size, width, depth):
            super(MLPNetwork, self).__init__()
            layers = [nn.Linear(input_size, width), nn.ReLU()]
            for _ in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(width, output_size))
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)
        
    model = MLPNetwork(input_size, output_size, int(config.get("width")), int(config.get("depth")))    

    return model
