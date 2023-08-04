import torch 
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import Network
from train import train_loop, test_loop

# define hyperparameters
BATCH_SIZE = 64
NUM_OF_EPOCHS = 10
LEARNING_RATE = 1e-3

def main():
    # create dataset and dataloader
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        download=True,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    test_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        download=True,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = BATCH_SIZE
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size = BATCH_SIZE
    )
    print('Data loading completed')

    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Currently using {device} as device')

    # define model
    model = Network().to(device)

    # define dictionary to store values to plot
    H = {
	"train_loss": [],
	"train_acc": [],
	"test_loss": [],
	"test_acc": []
    }

    # define loss function
    cross_entropy = nn.CrossEntropyLoss()

    # define optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # train and test
    for t in range(NUM_OF_EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, cross_entropy, optimizer, H)
        test_loop(test_loader, model, cross_entropy, H)
    print("Done processing model!")

    # plot model results
    with torch.no_grad():
        print('Plotting results \n-----------------------------')
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["test_loss"], label="test_loss")
        plt.plot(H["train_acc"], label="train_acc")
        plt.plot(H["test_acc"], label="test_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.show()
    

if __name__ == '__main__':
    main()

