import torch
from tqdm import tqdm
import time

def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):

    epoch_loss = 0.0
    count_samples = 0.0
    for idx, (input, target) in tqdm(enumerate(zip(batched_train_data, batched_train_label)), total=len(batched_train_data), desc="Training"):
        start_time = time.time()
        output, hidden = model(input)
        loss = torch.mean((output - target)**2)  # Mean Squared Error Loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Batch Time {batch_time:.3f} \t'
                   'Batch Loss {loss:.4f}\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time, loss=loss.item()))
    epoch_loss /= len(batched_train_data)

    if debug:
        print("* Average Loss of Epoch {} is: {:.4f}".format(epoch, epoch_loss))
    return epoch_loss


def evaluate(batched_test_data, batched_test_label, model, debug=True):

    epoch_loss = 0.0
    count_samples = 0.0
    for idx, (input, target) in tqdm(enumerate(zip(batched_test_data, batched_test_label)), total=len(batched_test_data), desc="Evaluating"):
        with torch.no_grad():
            output, hidden = model(input)
            loss = torch.mean((output - target)**2)  # Mean Squared Error Loss

        epoch_loss += loss.item()
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                   'Batch Loss {loss:.4f}\t').format(
                idx, len(batched_test_data), loss=loss.item()))
    epoch_loss /= len(batched_test_data)

    return epoch_loss
