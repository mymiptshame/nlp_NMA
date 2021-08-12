import torch.nn as nn


class ClassificationNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.linear_1 = nn.Linear(input_dim, 2000)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(2000, num_classes)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x
