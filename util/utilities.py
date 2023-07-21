import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import gc

# training function
def train(model, dataloader, criterion, optimizer, device='cpu', verbose=False):
    model.train()
    total_loss = 0.0
    untrainable_tensors = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        if verbose:
            print(f'\n\nForward pass on batch with size: {inputs.shape}')
        # forward pass
        outputs = model(inputs)

        if verbose:
            print(f'Model Output (shape: {outputs.shape}): {outputs}')
            print(f'Target (shape: {targets.shape}): {targets}')

        loss = criterion(outputs, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / (len(dataloader) - untrainable_tensors)
    return total_loss, avg_loss

# evaluation function
def evaluate(model, dataloader, criterion, device='cpu', verbose=False):
    model.eval()
    total_loss = 0.0
    unevaluable_tensors = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            try:
                if verbose:
                    print(f'\n\nEvaluating on batch with size: {inputs.shape}')

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

            except Exception as e:
                print(f'Unable to evaluate on tensor of size: {inputs.shape}')
                print('caugh an error, keep going ' + str(e))
                unevaluable_tensors += 1
                continue

            #if outputs.shape != targets.shape:
            #    targets = targets.unsqueeze(1)

            if verbose:
                print(f'Model Output (shape: {outputs.shape}): {outputs}')
                print(f'Target (shape: {targets.shape}): {targets}')

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / (len(dataloader) - unevaluable_tensors)
    return total_loss, avg_loss


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    return device