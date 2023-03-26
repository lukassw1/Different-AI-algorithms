from abc import ABC, abstractmethod
from functions import countourf_plot, plot_3d, g_gradeint
from functions import plot, f_gradeint
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""
    def __init__(self, start_point, gradient_function, beta, epsilon, max_loops) -> None:
        self.start_point = start_point
        self.gradient_function = gradient_function
        self.beta = beta
        self.epsilon = epsilon
        self.max_loops = max_loops

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return {
            "start_point": self.start_point,
            "gradient_function": self.gradient_function,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "max_loops": self.max_loops

        }

    def solve(self):
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...
        stops = [self.start_point]
        x = self.start_point
        loops = 0
        while True:
            gradient = self.gradient_function(x)
            x = x - self.beta*gradient
            stops.append(x)
            if np.linalg.norm(gradient) < self.epsilon or loops >= self.max_loops:
                minimum = x
                return minimum, stops
            loops += 1


def main():
    sol_f = Solver(1, f_gradeint, 0.8, 1e-7, 500000)
    minimum_f, stops_f = sol_f.solve()
    plot(stops_f, sol_f.beta)
    # sol_g = Solver(np.array([2, 2]), g_gradeint, 0.5, 1e-7, 5000)
    # minimum_g, stops_g = sol_g.solve()
    # plot_3d(stops_g, sol_g.beta)
    # countourf_plot(stops_g, sol_g.beta)


if __name__ == "__main__":
    main()