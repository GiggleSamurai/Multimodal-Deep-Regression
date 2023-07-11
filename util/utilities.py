import math
import time
import random

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm_notebook

RANDOM_SEED = 0

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def set_seed_nb():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED + 1)

def deterministic_init(net: nn.Module):
    for p in net.parameters():
        if p.data.ndimension() >= 2:
            set_seed_nb()
            nn.init.xavier_uniform_(p.data)
        else:
            nn.init.zeros_(p.data)

def train(model, dataloader, optimizer, criterion, scheduler=None, device='cpu'):
    model.train()
    total_loss = 0.
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    for batch_idx, data in enumerate(progress_bar):
        source = data[0].to(device)
        target = data[1].to(device)

        optimizer.zero_grad()
        prediction = model(source)

        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data[0].to(device)
            target = data[1].to(device)

            prediction = model(source)
            loss = criterion(prediction, target)

            total_loss += loss.item()
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss
