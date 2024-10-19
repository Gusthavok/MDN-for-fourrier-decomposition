import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        """
        input_size: Dimension of the input features
        hidden_layers: List where each element represents the number of neurons in that hidden layer
        output_size: Dimension of the output layer (number of classes or regression output)
        """
        super(MLP, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers[:-1]:  
            x = F.leaky_relu(layer(x))       
        
        x = self.layers[-1](x)
        return x

