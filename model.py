import torch
import torch.nn as nn 

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # LeNet Architecture
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
            
        self.lin_relu_stack = nn.Sequential(
            nn.Linear(in_features=800, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=10),
            nn.LogSoftmax(dim=1)
        )
        

    def forward(self, t):
        t = self.conv_relu_stack(t)
        t = torch.flatten(t, 1)
        out = self.lin_relu_stack(t)

        return out
    







