import random
import numpy as np
from copy import deepcopy


class Individual:
    def __init__(self, set_genes=[]):
        if set_genes:
            self.genes = set_genes
        else:
            self.genes = [random.randint(0, 1) for _ in range(200)]
        self.rating = None

    def mutate(self, pm):
        for gene in self.genes:
            if pm > random.uniform(0, 1):
                gene = gene ^ 1

    def __str__(self) -> str:
        return str(self.genes)


def rocket_landing(genes):
    current_v = 0
    height = 200
    fuel_amount = sum(genes)
    mass = fuel_amount + 200
    rate = - fuel_amount
    for gene in genes:
        current_a = -0.09
        if gene:
            mass -= 1
            current_a = current_a + 45/mass
        current_v += current_a
        height += current_v
        if height < 0:
            rate -= 1000
            break
        if height < 2 and abs(current_v) < 2:
            rate += 2000
            break
    return rate


def genetic_algorithm(f_q, start_population, pc, pm, t_max):
    t = 0
    population = start_population
    population = rating(start_population, f_q)
    x_best = find_best(population)
    o_best = x_best.rating
    print(str(x_best))
    print(o_best)
    while (t < t_max):
        population = selection(population)
        population = crossing(population, pc)
        population = mutating(population, pm)
        population = rating(population, f_q)
        x_t = find_best(population)
        if x_t.rating > o_best:
            x_best, o_best = x_t, x_t.rating
        t += 1
    return x_best, o_best


def crossing(cur_population, pc):
    new_population = []
    for i in range(0, len(cur_population), 2):
        if pc > random.uniform(0, 1):
            ind1 = cur_population[i]
            ind2 = cur_population[i+1]
            pivot = random.randint(0, 200)
            new_ind1 = deepcopy(ind1)
            new_ind2 = deepcopy(ind2)
            new_ind1.genes[pivot:] = ind2.genes[pivot:]
            new_ind2.genes[pivot:] = ind1.genes[pivot:]
        else:
            new_ind1 = cur_population[i]
            new_ind2 = cur_population[i+1]
        new_population.append(new_ind1)
        new_population.append(new_ind2)
    return new_population


def mutating(population, pm):
    for x in population:
        x.mutate(pm)
    return population


def rating(population, function_q):
    for x in population:
        x.rating = function_q(x.genes)
    return population


def find_best(population):
    x_bp = max(population, key=lambda x: x.rating)
    return x_bp


def selection(population):
    population = add_const(population)
    population_ratings = [x.rating for x in population]
    total = sum(population_ratings)
    population_probabilities = [x.rating/total for x in population]
    population = rem_const(population)
    next_population = np.random.choice(population, len(population), p=population_probabilities)
    return next_population


def add_const(population):
    for x in population:
        x.rating += 1200
    return population


def rem_const(population):
    for x in population:
        x.rating -= 1200
    return population


def make_population(amount):
    population = []
    for _ in range(amount):
        new_x = Individual()
        population.append(new_x)
    return population
