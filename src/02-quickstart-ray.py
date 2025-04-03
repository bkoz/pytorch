#
# Pytorch example with Ray
# This example demonstrates how to use Ray with PyTorch for distributed training.
# 
# Import necessary libraries
import ray
import ray.train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import time
import os
from mynets import NeuralNetwork
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 128

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    logger.info(f"Shape of X [N, C, H, W]: {X.shape}")
    logger.info(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"  # Force CPU for testing
logger.info(f"Using {device} device")

model = NeuralNetwork().to(device)
logger.info(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    logger.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Iinitialize Ray
ray.init()
# Create a Trainer

global_batch_size = 32
num_workers=2
use_gpu=False

train_config = {
    "lr": 1e-3,
    "epochs": 10,
    "batch_size_per_worker": global_batch_size // num_workers,
}

# Configure computation resources
scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

# Initialize a Ray TorchTrainer
trainer = TorchTrainer(
    train_loop_per_worker=train,
    train_loop_config=train_config,
    scaling_config=scaling_config,
    )

# [1] Prepare Dataloader for distributed training
# Shard the datasets among workers and move batches to the correct device
# =======================================================================
train_dataloader = ray.train.torch.prepare_data_loader(train_dataloader)
test_dataloader = ray.train.torch.prepare_data_loader(test_dataloader)

# [2] Prepare and wrap your model with DistributedDataParallel
# Move the model to the correct GPU/CPU device
# ============================================================
model = ray.train.torch.prepare_model(model)

epochs = 5
start = time.time()
for t in range(epochs):
    logger.info(f"Epoch {t+1}\n-------------------------------")

    if ray.train.get_context().get_world_size() > 1:
            # Required for the distributed sampler to shuffle properly across epochs.
        train_dataloader.sampler.set_epoch(t)

    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

end = time.time()
logger.info(f"Training time = {end - start:.2f} seconds")

# Save the model
dir_path = "models"
if os.path.isdir(dir_path):
    logger.info(f"Directory '{dir_path}' exists.")
else:
    logger.info(f"Directory '{dir_path}' does not exist.")
    os.mkdir("models")
torch.save(model.state_dict(), "models/model.pth")
logger.info("Saved PyTorch Model State to models/model.pth")

