"""
Definition of the mathematical problem for solving differential equations.

This file provides:
1) Abstract base classes to define and solve systems of differential equations.
2) Trial solution constructions for neural network approximations.
3) Numerical solutions for comparison using scipy's numerical solvers (Runge-Kutta).
"""

import abc 
from typing import Dict, Optional, List, Tuple, Any, Iterable, Union
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch as th
from scipy.integrate import solve_ivp #Numerical Method to compare - Explicit Runge-Kutta method of order 5
from scipy.interpolate import interp1d 


def compute_derivative(inputs: th.Tensor, outputs: th.Tensor) -> th.Tensor:
    """
    Computes the derivative of a given tensor with respect to its inputs.

    Args:
        inputs (th.Tensor): The input tensor (e.g., time points).
        outputs (th.Tensor): The output tensor (e.g., trial solution values).

    Returns:
        th.Tensor: The computed derivative of outputs with respect to inputs (e.g. d y_trial(x) /dx ).
    """
    return th.autograd.grad(outputs, inputs, grad_outputs=th.ones_like(outputs), create_graph=True)[0]

class SystemEquations(abc.ABC):
    """
    Abstract base class for defining and solving systems of first-order differential equations.

    Attributes:
        functions (List[str]): List of variable names (e.g., ["x", "y"]).
        domain (Tuple[float, float]): Time domain for the equations (start, end).
        initial_conditions (Dict[str, Tuple[float, float]]): Initial conditions for each variable.
    """
    def __init__(self, functions: List[str], domain: Tuple[float, float], initial_conditions: Dict[str, Tuple[float, float]]):
        """
        Initializes a system of first-order differential equations.

        Args:
            functions (List[str]): Names of the system variables.
            domain (Tuple[float, float]): Start and end points of the time domain.
            initial_conditions (Dict[str, Tuple[float, float]]): Initial conditions for each variable.
        """
        self.functions = functions
        self.domain = domain
        self.initial_conditions = initial_conditions

    def configuration(self) -> Dict[str, Any]:
        """
        Returns the configuration of the system.

        Returns:
            Dict[str, Any]: A dictionary containing the system configuration, including:
                - "functions": Variable names.
                - "domain": Time domain range.
                - "initial_conditions": Initial conditions for each variable.
        """
        return {
            "functions": self.functions,
            "domain": self.domain,
            "initial_conditions": self.initial_conditions,
        }

    def calculate_trial_solution(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Constructs trial solutions that satisfy initial conditions.

        Args:
            inputs (th.Tensor): Input tensor (time values).
            outputs (Dict[str, th.Tensor]): Neural network outputs.

        Returns:
            Dict[str, th.Tensor]: Dictionary containing trial solutions for each function.
        """
        return {
           function: self.initial_conditions[function][1] + (inputs / self.domain[1] - self.initial_conditions[function][0]) * outputs[function]
            for function in self.functions
        }
    
    @abc.abstractmethod
    def calculate_loss(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> th.Tensor:
        """
        Computes the loss function based on the system dynamics.

        Args:
            inputs (th.Tensor): Input tensor (e.g., time points).
            outputs (Dict[str, th.Tensor]): Neural network outputs for each function.

        Returns:
            th.Tensor: Computed loss value.
        """
        raise NotImplementedError

    def system(self, t: float, y: Union[np.ndarray, Iterable, int, float]) -> Union[np.ndarray, Iterable, int, float]:
        """
        Defines the system of first-order differential equations.

        Args:
            t (float): Time variable.
            y (Union[np.ndarray, Iterable, int, float]): State variables.

        Returns:
            Union[np.ndarray, Iterable, int, float]: The computed derivatives.
        """
        raise NotImplementedError

    def solve_numerically(self, inputs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Solves the system of equations numerically using the `solve_ivp` function.

        Args:
            inputs (np.ndarray): Time points where the solution is evaluated.

        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of numerical solutions for each variable.
        """
        try:
            sol = solve_ivp(
                self.system,
                self.domain,
                [self.initial_conditions[var][1] for var in self.functions],
                t_eval=inputs
            )
            return {
                function: sol.y[i]
                for i, function in enumerate(self.functions)
            }
        except NotImplementedError:
            return None


class SecondOrderEquations(abc.ABC):
    """
    Abstract base class for solving second-order differential equations.

    Attributes:
        function (str): The dependent variable (e.g., "x").
        domain (Tuple[float, float]): Start and end points of the time domain.
        initial_conditions (Dict[str, Tuple[float, float]]): Initial conditions for the function and its derivative.
    """
    def __init__(self, function: str, domain: Tuple[float, float], initial_conditions: Dict[str, Tuple[float, float]], boundary_type: str = "pvi"):
        """
        Initializes a second-order differential equation.

        Args:
            function (str): Name of the dependent variable.
            domain (Tuple[float, float]): Start and end points of the time domain.
            initial_conditions (Dict[str, Tuple[float, float]]): Initial conditions for the function and its derivative.
            boundary_type (str, optional): Type of boundary conditions to enforce ("pvi" or "dirichlet"). Default is "pvi".
        """
        self.function = function
        self.functions = [function]  # Keep as list for compatibility
        self.domain = domain
        self.initial_conditions = initial_conditions
        self.boundary_type = boundary_type  # Store boundary type

    def configuration(self) -> Dict[str, Any]:
        """
        Returns the configuration of the second-order equation.

        Returns:
            Dict[str, Any]: A dictionary containing the equation configuration.
        """
        
        return {
            "function": self.function,
            "domain": self.domain,
            "initial_conditions": self.initial_conditions,
        }

    def calculate_trial_solution(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Constructs the trial solution for second-order differential equations.

        Args:
            inputs (th.Tensor): Input tensor (spatial or time values).
            outputs (Dict[str, th.Tensor]): Neural network outputs.

        Returns:
            Dict[str, th.Tensor]: Dictionary containing trial solutions for the function and its first derivative.
        """
        
        if not inputs.requires_grad:
            inputs = inputs.clone().detach().requires_grad_(True)

        # Neural network's output is N(x, params)
        N_x = outputs[self.function]
        
        if self.boundary_type == "pvi": # Initial Value Problem (Cauchy problem)
            x0 = self.initial_conditions["x"][1]  
            y0 = self.initial_conditions["y"][1]  
            t0 = self.initial_conditions["x"][0]  
            x_trial = x0 + y0 * (inputs - t0) + (inputs - t0)**2 * N_x

        elif self.boundary_type == "dirichlet": # Dirichlet Boundary Conditions
            a, b = self.domain  
            y_a = self.initial_conditions["x"][1]  
            y_b = self.initial_conditions["y"][1]  
            x_trial = y_a * ((b - inputs) / (b - a)) + y_b * ((inputs - a) / (b - a)) + ((inputs - a) * (b - inputs) * N_x)
            #print(a,b,y_a,y_b,x_trial)

        else:
            raise ValueError("Invalid boundary_type. Use 'pvi' for Initial Value Problem or 'dirichlet' for Dirichlet conditions.")
        
        # Compute the derivative for x'(t)
        x_dot_trial = compute_derivative(inputs, x_trial)

        return {"x": x_trial, "y": x_dot_trial}

    @abc.abstractmethod
    def calculate_loss(self, inputs: th.Tensor, outputs: Dict[str, th.Tensor]) -> th.Tensor:
        raise NotImplementedError

    def solve_numerically(self, inputs: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Solves the second-order differential equation numerically.

        Args:
            inputs (np.ndarray): Array of time points where the solution is evaluated.

        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary of numerical solutions for "x" and "y" (x' if PVI, dx/dt if Dirichlet).
        """
        try:
            inputs_np = inputs.view(-1).cpu().numpy() if isinstance(inputs, th.Tensor) else inputs

            if self.boundary_type == "pvi":
                # Solve as an initial value problem using solve_ivp
                def system(t, z):
                    x, y = z  # y = dx/dt
                    dxdt = y
                    dydt = self.equation(x, y, t)  # Second-order ODE
                    return [dxdt, dydt]

                sol = solve_ivp(
                    system,
                    self.domain,
                    [self.initial_conditions["x"][1], self.initial_conditions["y"][1]],
                    t_eval=inputs_np
                )

                real_values = {
                    "x": interp1d(sol.t, sol.y[0], kind='cubic', fill_value="extrapolate")(inputs_np),
                    "y": interp1d(sol.t, sol.y[1], kind='cubic', fill_value="extrapolate")(inputs_np),
                }

            elif self.boundary_type == "dirichlet":
                # Solve as a boundary value problem using solve_bvp
                from scipy.integrate import solve_bvp

                a, b = self.domain  # Boundary points
                y_a = self.initial_conditions["x"][1]  # x(a)
                y_b = self.initial_conditions["y"][1]  # x(b)

                def system(t, Y):
                    x, y = Y
                    dxdt = y
                    dydt = self.equation(x, y, t)
                    return np.vstack((dxdt, dydt))

                def bc(Y_a, Y_b):
                    return np.array([Y_a[0] - y_a,  # x(a) = y_a
                                     Y_b[0] - y_b])  # x(b) = y_b

                # Initial guess for solution
                t_guess = np.linspace(a, b, 100)
                x_guess = np.linspace(y_a, y_b, 100)  # Linear interpolation
                y_guess = np.zeros_like(x_guess)  # Assume x' ≈ 0 initially
                Y_guess = np.vstack((x_guess, y_guess))

                sol = solve_bvp(system, bc, t_guess, Y_guess)

                real_values = {
                    "x": interp1d(sol.x, sol.y[0], kind='cubic', fill_value="extrapolate")(inputs_np),
                    "y": interp1d(sol.x, sol.y[1], kind='cubic', fill_value="extrapolate")(inputs_np),
                }

            else:
                raise ValueError("Invalid boundary_type. Use 'pvi' for Initial Value Problems or 'dirichlet' for Boundary Value Problems.")

            return real_values

        except NotImplementedError:
            return None