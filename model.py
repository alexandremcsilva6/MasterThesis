"""
Training, Evaluation, and Integration of Neural Networks for Solving Differential Equations.

This module performs:
1. Training neural networks to approximate solutions of ODEs and systems of ODEs.
2. Evaluating performance by comparing predictions to numerical solutions (Runge-Kutta).
3. Handling advanced training techniques such as adaptive sampling and loss visualization.
"""

import json
import os.path
import random
from datetime import datetime
from itertools import combinations
from typing import List, NamedTuple, Optional, Dict, Callable, Tuple
import numpy as np
import torch as th
from matplotlib import pyplot as plt
from torchinfo import summary
from tqdm import tqdm
from scipy.stats import skewnorm

from equations import SecondOrderEquations, compute_derivative, SystemEquations
from modules import EquationsModel

th.set_default_dtype(th.float64) # Set PyTorch's default tensor type to float64 for higher precision

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def assert_directory_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def set_all_seeds(seed: int):
    """
    Sets a fixed random seed for all possible sources of randomness to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


class Configuration(NamedTuple):
    """
    A configuration container for neural network settings.

    Attributes:
        seed (int): Random seed for reproducibility.
        features (List[int]): List defining the number of neurons per hidden layer.
        activation_function (Callable): Activation function used in the neural network.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        steps (int): Number of discrete time steps used in training.
    """
    seed: int
    features: List[int]
    activation_function: Callable[[th.Tensor], th.Tensor]
    learning_rate: float
    epochs: int
    steps: int


def calculate_y_limits(ylim: Tuple[float, float], losses: List[float], z_threshold: float) -> Tuple[float, float]:
    """
    Auxiliary function to adjust the y-axis limits for loss plots by filtering out extreme outliers.

    Args:
        ylim (Tuple[float, float]): The current y-axis limits.
        losses (List[float]): A list of recorded loss values.
        z_threshold (float): The threshold for filtering out high-loss values.

    Returns:
        Tuple[float, float]: Adjusted y-axis limits.
    """
    bottom, top = ylim
    losses = np.array(losses)
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    if std_loss < 1e-8:
        return min(losses) - 0.0005, np.max(losses) if len(losses) > 0 else top
        
    z_scores = (losses - mean_loss) / std_loss
    filtered_losses = losses[z_scores <= z_threshold]
    
    if filtered_losses.size == 0:
        return max(bottom, 0), top

    return min(filtered_losses) - 0.0005, np.max(filtered_losses)


class Model:
    """
    Manages training, evaluation, and testing of neural networks for solving differential equations.

    Attributes:
        configuration (Configuration): Stores neural network and training settings.
        system_equations (SystemEquations): Defines the differential equations to solve.
        model (EquationsModel): The neural networks used to approximate solutions.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """
    
    def __init__(
        self,
        name: str,
        configuration: Configuration,
        system_equations: SystemEquations,
    ):
        
        self.configuration = configuration
        set_all_seeds(configuration.seed)
        self.output_path = os.path.join(OUTPUT_PATH, name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.system_equations = system_equations
        
        # Initialize the neural network model
        self.model = EquationsModel(system_equations.functions, configuration.features, configuration.activation_function)
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=configuration.learning_rate)
        
        #print(self.model)

        print("Summary completed.")

    def load(self, model_path: str, optimizer_path: str):
        """
        Loads a pre-trained model and optimizer state.
         
        Args:
        model_path (str): Path to the saved model weights.
        optimizer_path (str): Path to the saved optimizer state.
        """
        self.model.load_state_dict(th.load(model_path, weights_only=True))
        self.optimizer.load_state_dict(th.load(optimizer_path, weights_only=True))

    def _save_train(self, losses: List[float]):
        """
        Saves the trained model, optimizer state, training configuration, and loss curve.

        Args:
            losses (List[float]): List of loss values recorded during training.
        """
        assert_directory_exists(self.output_path)

        th.save(self.model.state_dict(), os.path.join(self.output_path, "model.pt"))
        th.save(self.optimizer.state_dict(), os.path.join(self.output_path, "optimizer.pt"))

        #with open(os.path.join(self.output_path, "configuration.json"), "w") as f:
        #    configuration = self.configuration._asdict()
        #    configuration["system_equations"] = self.system_equations.configuration()
        #    configuration["system_equations"]["name"] = self.system_equations.__class__.__name__
        #    configuration["activation_function"] = configuration["activation_function"].__class__.__name__
        #    f.write(json.dumps(configuration, indent=4))

        with open(os.path.join(self.output_path, "configuration.json"), "w") as f:
            configuration = self.configuration._asdict()
            configuration["system_equations"] = self.system_equations.configuration()
            configuration["system_equations"]["name"] = self.system_equations.__class__.__name__
            configuration["activation_function"] = configuration["activation_function"].__class__.__name__

            # ✅ Convert PyTorch tensors to Python floats for JSON serialization
            for key, value in configuration["system_equations"]["initial_conditions"].items():
                configuration["system_equations"]["initial_conditions"][key] = (value[0], float(value[1]))

            f.write(json.dumps(configuration, indent=4))


        fig, axes = plt.subplots(3, 1, figsize=(12, 6 * 3))
        for i in range(3):
            axes[i].plot(losses, label="Training Loss")
            axes[i].set_title("Loss During Training")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Loss")
            
            axes[i].set_ylim(calculate_y_limits(axes[i].get_ylim(), losses, 2 - i))
            axes[i].grid()
            axes[i].legend()

        plt.savefig(os.path.join(self.output_path, "loss.png"))
        plt.show()

    def train(self):
        """
        Trains the neural network to approximate the solution of the differential equation.
        """
        
        inputs = th.linspace(
            self.system_equations.domain[0],
            self.system_equations.domain[1],
            self.configuration.steps
        ).view(-1, 1).requires_grad_()

        losses: List[float] = []

        for _ in tqdm(range(self.configuration.epochs)):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.system_equations.calculate_loss(inputs, outputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            # Print loss every 50 epochs
            if (_ + 1) % 50 == 0:  # Adding 1 to start from epoch 1 instead of 0
                print(f"Epoch {_+1}/{self.configuration.epochs}, Loss: {loss.item():.6f}")

        self._save_train(losses)

    def train_weighted_density(self, weights: List[int]):
        """
        Trains the neural network using a weighted density approach.
        Instead of uniformly sampling points in the domain, this method divides the domain into intervals, 
        and assigns to each sub-interval a proportion of training points.

        Args:
            weights (List[int]): A list of weights corresponding to intervals within the domain.
                                 Higher weights result in more points being sampled in that region.
    """
        domain_start, domain_end = self.system_equations.domain[0], self.system_equations.domain[1]
        num_intervals = len(weights)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        interval_edges = th.linspace(domain_start, domain_end, num_intervals + 1)
        
        sampled_points = []
        for i in range(num_intervals):
            interval_start = interval_edges[i].item()
            interval_end = interval_edges[i + 1].item()

            # Number of points in this interval is proportional to the weight
            num_points = int(normalized_weights[i] * self.configuration.steps)
            points = th.linspace(interval_start, interval_end, num_points)
            sampled_points.append(points)

        # Concatenate points into a single tensor
        inputs = th.cat(sampled_points).view(-1, 1).requires_grad_()

        losses: List[float] = []

        for _ in tqdm(range(self.configuration.epochs)):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.system_equations.calculate_loss(inputs, outputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        self._save_train(losses)

    def train_located_density(self,high_density_point: float, skewness: float):
        """
        Trains the neural network with a probability density centered around a specified region.
        This method generates a skewed normal distribution of points and concentrates more points around a high_density_point.

        Args:
            high_density_point (float): The center point where the training data should be concentrated.
            skewness (float): Controls the asymmetry of the distribution (<0 to the left, >0 to the right, =0 symmetric)
        """
        
        domain_start, domain_end = self.system_equations.domain[0], self.system_equations.domain[1]
        steps = self.configuration.steps
        # Ensure the high_density_point is within the domain
        if not (domain_start <= high_density_point <= domain_end):
            raise ValueError("high_density_point must be within the domain range.")

        x = np.linspace(domain_start, domain_end, steps)
        skewed_pdf = skewnorm.pdf(x, a=skewness, loc=high_density_point, scale=(domain_end - domain_start) / 4)
        skewed_pdf /= np.sum(skewed_pdf)
        sampled_points = np.random.choice(x, size=steps, p=skewed_pdf)
        inputs = th.tensor(sampled_points, dtype=th.float64).view(-1, 1).requires_grad_()
        
        losses: List[float] = []

        for _ in tqdm(range(self.configuration.epochs)):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.system_equations.calculate_loss(inputs, outputs)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        self._save_train(losses)

    
    def train_adaptive(self, percentage = 1.0, resampling_factor = 0.1):
        """
        Trains the neural network using **adaptive sampling**.
        Unlike uniform sampling, this method identifies regions where the model struggles, 
        by analysing gradints, and focuses on high gradient regions.

        Args:
            percentage (float, optional): The percentage of high-gradient points to retain.
            resampling_factor (float, optional): The proportion of additional points to be added in high-gradient regions.
        """
        inputs = th.linspace(
            self.system_equations.domain[0],
            self.system_equations.domain[1],
            self.configuration.steps).view(-1, 1).requires_grad_()

        losses: List[float] = []

        for haha in tqdm(range(self.configuration.epochs)):
            self.optimizer.zero_grad()
            #print(inputs.shape)
            outputs = self.model(inputs)
            loss = self.system_equations.calculate_loss(inputs, outputs)
            inputs.retain_grad()  # Retain gradients for inputs
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
            gradients = inputs.grad.abs().view(-1)
            
            num_points = int(len(gradients) * (percentage / 100.0))
            _, top_indices = gradients.topk(num_points)
            high_gradient_inputs = inputs[top_indices]
            additional_inputs = th.linspace(
                high_gradient_inputs.min(),
                high_gradient_inputs.max(),
                int(num_points * resampling_factor)
            ).view(-1, 1)
            
            inputs = th.cat((inputs, additional_inputs)).requires_grad_().view(-1, 1)

        self._save_train(losses)

    def eval(self, inputs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluates the trained neural network by computing trial solutions.
        Performs a forward pass in the networks and compute the trial solutions.

        Args:
            inputs (np.ndarray): Array of time points at which the solution should be evaluated.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing "x": Predicted solution 'x(t)' and 
                                                         "y": If second-order, 'x'(t)', if system 'y(t)'.
        """
        
        tensor_inputs = th.tensor(inputs, dtype=th.float64).view(-1, 1).requires_grad_()
        outputs = self.model(tensor_inputs)
        trial_solutions = self.system_equations.calculate_trial_solution(tensor_inputs, outputs)

        if isinstance(self.system_equations, SecondOrderEquations):
            trial_solutions["y"] = compute_derivative(tensor_inputs, trial_solutions["x"])
        
        return {function: trial_solution.detach().numpy() for function, trial_solution in trial_solutions.items()}

    def test(self, inputs: Optional[np.ndarray] = None):
        """
        Tests the trained neural network by comparing it against a numerical solution.

        This method computes the numerical solution, evaluates the neural network's predictions and 
        plots both functions against time and phase space. It stores everything.

    Args:
        inputs (Optional[np.ndarray]): Array of time points for evaluation. If 'None', a uniform grid is used.
    """
        if inputs is None:
            inputs = np.linspace(
                self.system_equations.domain[0],
                self.system_equations.domain[1],
                self.configuration.steps
            )

        predicted_values = self.eval(inputs)
        #print(predicted_values["x"])

        #predicted_values = self.eval(inputs)

        # ✅ Corrected keys based on the double pendulum system
        #print(predicted_values["x1"])  # First pendulum angle
        #print(predicted_values["x2"])  # Second pendulum angle

        real_values = self.system_equations.solve_numerically(inputs)
        print(real_values)
        
        assert_directory_exists(self.output_path)

        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
'''
        # ✅ First pendulum angle
        axes[0].plot(inputs, predicted_values["x1"], label="NN Approximation: x1(t)", linestyle="--")
        if real_values is not None:
            axes[0].plot(inputs, real_values["x1"], label="Numerical Solution: x1(t)")
        
        # ✅ Second pendulum angle
        axes[1].plot(inputs, predicted_values["x2"], label="NN Approximation: x2(t)", linestyle="--")
        if real_values is not None:
            axes[1].plot(inputs, real_values["x2"], label="Numerical Solution: x2(t)")
        
        axes[0].set_title("Angle of First Pendulum (x1)")
        axes[0].set_xlabel("Time t")
        axes[0].set_ylabel("x1(t)")
        axes[0].legend()
        axes[0].grid()
        
        axes[1].set_title("Angle of Second Pendulum (x2)")
        axes[1].set_xlabel("Time t")
        axes[1].set_ylabel("x2(t)")
        axes[1].legend()
        axes[1].grid()
'''


        
        axes[0].plot(inputs, predicted_values["x"], label="NN Approximation: x(t)", linestyle="--")
        if real_values is not None:
            axes[0].plot(inputs, real_values["x"], label="Numerical Solution: x(t)")
        axes[0].set_title("Neural Network vs Numerical Solution")
        axes[0].set_xlabel("Time t")
        axes[0].set_ylabel("x(t)")
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(predicted_values["x"], predicted_values["y"], label="NN Phase Space", linestyle="--")
        if real_values is not None:
            axes[1].plot(real_values["x"], real_values["y"], label="Numerical Phase Space")
        axes[1].set_title("Phase Space: x vs x'")
        axes[1].set_xlabel("x(t)")
        axes[1].set_ylabel("x'(t)")
        axes[1].grid()
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "solution.png"))
        plt.show()