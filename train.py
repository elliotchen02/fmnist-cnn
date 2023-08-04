import torch 
import torch.nn as nn
from torch.utils.data import DataLoader



def train_loop(dataloader, model, loss_fn, optimizer, logger):
    # Unnecessary here but would be needed if batch normalization were used
    model.train()

    # For plotting purposes
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        # Calculate loss
        prediction = model(X)
        loss = loss_fn(prediction, y)
        total_loss += loss

        # Backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Update logger
    avg_train_loss = total_loss / num_batches
    correct /= size
    logger["train_loss"].append(avg_train_loss.detach().numpy())
    logger["train_acc"].append(correct)


def test_loop(dataloader, model, loss_fn, logger):
    # Unnecessary here unless batch normalization
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss, test_loss, correct = 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            prediction = model(X)
            test_loss += loss_fn(prediction, y).item()
            avg_loss += loss_fn(prediction, y)
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # Update logger
    avg_loss /= num_batches
    logger["test_loss"].append(avg_loss.detach().numpy())
    logger["test_acc"].append(correct)


