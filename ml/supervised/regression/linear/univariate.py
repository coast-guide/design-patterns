import numpy as np
import logging
from typing import Tuple, Callable, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_cost(w: float, b: float, x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the mean squared error cost for linear regression.

    Args:
        w (float): Weight (slope) of the linear model.
        b (float): Bias (intercept) of the linear model.
        x (np.ndarray): Input feature data (independent variable).
        y (np.ndarray): Output target data (dependent variable).

    Returns:
        float: The computed cost (mean squared error).
    """
    m = x.shape[0]  # Number of training examples
    total_cost = np.sum(((w * x) + b - y) ** 2)
    return total_cost / (2 * m)


def compute_gradients(w: float, b: float, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Computes the gradient of the cost function with respect to `w` (weight) and `b` (bias).

    Args:
        w (float): Weight of the linear model.
        b (float): Bias of the linear model.
        x (np.ndarray): Input feature data (independent variable).
        y (np.ndarray): Output target data (dependent variable).

    Returns:
        Tuple[float, float]: The gradients with respect to `w` and `b`.
    """
    m = x.shape[0]  # Number of training examples
    error = (w * x + b) - y
    dj_dw = np.sum(error * x) / m  # Partial derivative w.r.t. w
    dj_db = np.sum(error) / m      # Partial derivative w.r.t. b
    return dj_dw, dj_db


def gradient_descent(
        w: float,
        b: float,
        x: np.ndarray,
        y: np.ndarray,
        alpha: float,
        num_iterations: int,
        cost_function: Callable[[float, float, np.ndarray, np.ndarray], float],
        gradient_function: Callable[[float, float, np.ndarray, np.ndarray], Tuple[float, float]],
        tolerance: float = 1e-6
    ) -> Tuple[float, float, List[Tuple[float, float]], List[float]]:
    """
    Performs gradient descent to optimize `w` (weight) and `b` (bias) for linear regression.
    Includes early stopping if the improvement in cost falls below the given tolerance.

    Args:
        w (float): Initial weight of the linear model.
        b (float): Initial bias of the linear model.
        x (np.ndarray): Input feature data.
        y (np.ndarray): Output target data.
        alpha (float): Learning rate for gradient descent.
        num_iterations (int): Maximum number of iterations for gradient descent.
        cost_function (Callable): Function to calculate the cost (error).
        gradient_function (Callable): Function to compute the gradients.
        tolerance (float): The threshold to trigger early stopping when the cost difference is small.

    Returns:
        Tuple[float, float, List[Tuple[float, float]], List[float]]:
            - Optimized values of `w` and `b`
            - History of parameters during training
            - History of cost values during training
    """
    parameter_history = [(w, b)]
    cost_history = [cost_function(w, b, x, y)]

    for iteration in range(num_iterations):
        dj_dw, dj_db = gradient_function(w, b, x, y)

        # Update parameters using gradient descent rule
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Compute the new cost
        current_cost = cost_function(w, b, x, y)
        cost_history.append(current_cost)
        parameter_history.append((w, b))

        # Log the progress
        logging.info(f"Iteration {iteration + 1}: w = {w}, b = {b}, cost = {current_cost}")

        # Early stopping condition
        if abs(cost_history[-2] - current_cost) < tolerance:
            logging.info(f"Early stopping at iteration {iteration + 1} due to convergence.")
            break

    return w, b, parameter_history, cost_history


if __name__ == "__main__":
    # Example usage with synthetic data
    logging.info("Starting uni-variate linear regression with gradient descent")

    # Generate synthetic data
    x_train = np.array([1, 2, 3, 4, 5])
    y_train = np.array([2, 4, 6, 8, 10])  # Target is y = 2x

    # Hyperparameters
    w_init = 0.0
    b_init = 0.0
    learning_rate = 0.01
    iterations = 1000
    tolerance = 1e-6

    # Perform gradient descent
    w_final, b_final, parameters, cost_values = gradient_descent(
        w_init,
        b_init,
        x_train,
        y_train,
        learning_rate,
        iterations,
        calculate_cost,
        compute_gradients,
        tolerance
    )

    logging.info(f"Training complete. Final weight: {w_final}, Final bias: {b_final}")
