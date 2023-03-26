from abc import ABC, abstractmethod
from genetic_functions import rating, find_best, selection, crossing, mutating, make_population, rocket_landing
import matplotlib.pyplot as plt
import numpy as np


class GeneticSolver(ABC):
    """A solver. It may be initialized with some hyperparameters."""
    def __init__(self, t_max, population_size):
        self.t_max = t_max
        self.population_size = population_size

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        return {
            "t_max": self.t_max,
            "population_size": self.population_size
        }

    def solve(self, pc, pm, function_q, population=[]):
        """
        A method that solves the given problem for given initial solutions.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        if not population:
            start_population = make_population(self.population_size)
        t = 0
        population = start_population
        population = rating(start_population, function_q)
        x_best = find_best(population)
        o_best = x_best.rating
        while (t < self.t_max):
            population = selection(population)
            population = crossing(population, pc)
            population = mutating(population, pm)
            population = rating(population, function_q)
            x_t = find_best(population)
            if x_t.rating > o_best:
                x_best, o_best = x_t, x_t.rating
            t += 1
        return x_best, o_best


def main():
    pcs = [0.3, 0.4, 0.5, 0.6, 0.8, 0.9]
    solver = GeneticSolver(500, 100)
    pcs_data = []
    for pc in pcs:
        pc_data = []
        for _ in range(25):
            x, y = solver.solve(pc, 0.01, rocket_landing)
            pc_data.append(y)
        pcs_data.append(pc_data)
    make_box_plot(pcs_data, pcs)
    avgs = [sum(x)/len(x) for x in pcs_data]
    print("Results for pm = 0.01, t_max = 500, population_size = 100")
    for x in range(len(pcs)):
        print(f"{pcs[x]}: {avgs[x]}\n")


def make_box_plot(ratings_data, y_data):
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(ratings_data, patch_artist=True, notch='True', vert=0)
    ax.set_xlabel("ratings for each pc")
    ax.set_yticklabels(y_data)
    ax.set_xlim(1900, 1950)
    plt.title("Result for pm = 0.01, t_max = 500, population_size = 100")
    plt.savefig("box_plot.png")


if __name__ == "__main__":
    main()
