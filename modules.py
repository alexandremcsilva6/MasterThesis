"""
Neural Network Architecture for Approximating Solutions to Differential Equations.

This module provides reusable neural network components for approximating solutions to ordinary differential equations (ODEs). 

It defines:
1. 'LinearModel': A simple feedforward neural network that approximates a single function.
2. 'EquationsModel': A collection of 'LinearModels', where each network approximates a different function in a (possible) system of ODEs.
"""

from typing import List, Dict, Callable
import torch as th
from torch import nn


class LinearModel(nn.Module):
    """
    A feedforward neural network.

    Args:
        features (List[int]): A list specifying the number of neurons in each hidden layer. 
                              The length of this list determines the number of hidden layers.
        activation_function (Callable): Activation function to use in the hidden layers.
    """
    
    def __init__(self, features: List[int], activation_function: Callable[[th.Tensor], th.Tensor], *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = []
        input_dim = 1 #Inputs are scalars for our problem

        #Create fully connected hidden layers with the defined activation function
        for output_dim in features:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation_function)
            input_dim = output_dim
        
        #Final output layer (no activation function) and combine all layers into a sequential model
        layers.append(nn.Linear(input_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, t):
        """
        Forward pass through the neural network.

        Args:
            t (th.Tensor): Input tensor (e.g., time t).
        
        Returns:
            th.Tensor: Predicted value of the function.
        """
        return self.layers(t)

'''
class EquationsModel(nn.Module):
    """
    A collection of neural networks, where each network approximates a separate function.

    Args:
        functions (List[str]): List of function names (e.g., ['x', 'y'] for systems of two equations).
        features (List[int]): Number of neurons in each hidden layer for all networks.
        activation_function (Callable): Activation function to use between layers.
    """
    def __init__(
        self,
        functions: List[str],
        features: List[int],
        activation_function: Callable[[th.Tensor], th.Tensor],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        layers = []
        input_dim = 1 #Inputs are scalars for our problem

        #Create fully connected hidden layers with the defined activation function
        for output_dim in features:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(activation_function)
            input_dim = output_dim
        
        #Final output layer (no activation function) and combine all layers into a sequential model
        layers.append(nn.Linear(input_dim, 2))
        self.layers = nn.Sequential(*layers)
        
        # Create a dictionary of neural networks, one for each function
        self.equations = {
            function: function
            for function in functions
        }

    def forward(self, t: th.Tensor):
        """
        Forward pass through all neural networks.

        Args:
            t (th.Tensor): Input tensor (e.g., time t).
        
        Returns:
            Dict[str, th.Tensor]: Predicted values for each function.
                                  Keys correspond to function names (e.g., 'x', 'y').
        """
        outputs = self.layers(t)
        print(outputs.shape, self.equations)
        return {
            function: outputs[:, i]
            for i, function in enumerate(self.equations.keys())
        }
'''

class EquationsModel(nn.Module):
    """
    A collection of neural networks, where each network approximates a separate function.

    Args:
        functions (List[str]): List of function names (e.g., ['x', 'y'] for systems of two equations).
        features (List[int]): Number of neurons in each hidden layer for all networks.
        activation_function (Callable): Activation function to use between layers.
    """
    def __init__(
        self,
        functions: List[str],
        features: List[int],
        activation_function: Callable[[th.Tensor], th.Tensor],
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # Create a dictionary of neural networks, one for each function
        self.equations = nn.ModuleDict({
            function: LinearModel(features, activation_function)
            for function in functions
        })

    def forward(self, t: th.Tensor):
        """
        Forward pass through all neural networks.

        Args:
            t (th.Tensor): Input tensor (e.g., time t).
        
        Returns:
            Dict[str, th.Tensor]: Predicted values for each function.
                                  Keys correspond to function names (e.g., 'x', 'y').
        """
        return {
            function: model(t)
            for function, model in self.equations.items()
        }
