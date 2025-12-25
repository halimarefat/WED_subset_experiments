import torch.nn as nn


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer):
        super().__init__()
        layers = [nn.Linear(input_size, neurons_per_layer[0]), nn.ReLU()]
        for i in range(1, hidden_layers):
            layers.append(nn.Linear(neurons_per_layer[i - 1], neurons_per_layer[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(neurons_per_layer[-1], output_size))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
