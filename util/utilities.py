import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import gc

# training function
def train(model, dataloader, criterion, optimizer, device='cpu'):
    model.train()
    total_loss = 0.0
    for inputs, targets in dataloader:
        #inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        gc.collect()

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss

# evaluation function
def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = inputs.to(torch.float32), targets.to(torch.float32)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            gc.collect()

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss
