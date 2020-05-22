import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import *
import time


items = np.array([(5,3), (6,2), (1,4), (9,5), (2,8), (8,9), (4,10), (3,1), (7,6), (10,7)])
CAPACITY = 20
NUM_GENES = len(items)

# TODO:
# Global variables (specific to the problem at hand)
# TODO: make these modifiable by the user (A GA_Problem class?)

def random_individual(length=NUM_GENES):
    """
    Generates a random individual with number of genes of a given length.
    Each gene is either a 0 or 1.
    """
    return np.random.randint(low = 0, high = 2, size = length)

def random_population(population_size=100, length=10, vectorized=False):
    """
    Generates a population with `population_size` number of individuals.
    Each individual will have `length` number of genes.
    """
    list_of_individuals = [random_individual(length) for i in range(population_size)]
    if vectorized:
        return np.array(list_of_individuals)
    else:
        return list_of_individuals

def evaluate_individual(individual):
    """
    Given an individual, returns the total benefit and total volume.
    """
    total_benefit = sum(individual * items[:, 0])
    total_volume = sum(individual * items[:, 1])
    return total_benefit, total_volume


def mutate(individual):
    """
    Given an individual, returns a new individual with a random gene flipped.
    Does not modify the original individual.
    """
    individual = deepcopy(individual)
    index = random.randint(0,len(individual)-1)
    if individual[index] == 0:
        individual[index] = 1
    else:
        individual[index] = 0
    return individual

def mutate_v2(individual, p=1):
    """
    Flips each gene with probability p.
    Returns a new individual.
    """
    individual = deepcopy(individual)
    for i in range(len(individual)):
        roll = random.random()
        if roll <= p:
            if individual[i] == 0:
                individual[i] = 1
            else:
                individual[i] = 0
    return individual

def mutate_population(population):
    """
    Given a population, returns a new population where each gene is mutated.
    Does not modify the original population.
    """
    new_population = []
    for individual in population:
        new_population.append(mutate(individual))
    return new_population

def fitness(individual):
    """
    Evaluates an individual's fitness level.
    """
    benefit, volume = evaluate_individual(individual)
    if volume > 20:
        fitness = benefit - (4)*abs(volume - CAPACITY)
    else:
        fitness =  benefit - abs(volume - CAPACITY)
    return fitness

# ### Random Search
# * Generate individuals randomly, return the one with the highest fitness.
# * Just a Baseline model
def random_search(epochs = 1000):
    start_time = time.time()

    best_individual = None
    best_benefit = 0

    for i in range(epochs):
        individual_i = random_individual()
        benefit_i, volume_i = evaluate_individual(individual_i)
        if benefit_i > best_benefit and volume_i <= CAPACITY:
            best_benefit = benefit_i
            best_individual = individual_i

    print("Best individual: ", best_individual)
    best_benefit, best_volume = evaluate_individual(best_individual)
    print("Benefit: ", best_benefit)
    print("Volume: ", best_volume)

    print("Random search took", str(time.time() - start_time)[0:6], "seconds to run")



### Brute Force
# The best solution turns out to have:
# * **Benefit**: 33
# * **Volume**: 18
def brute_force():
    """
    HEY
    """
    start_time = time.time()

    all_solutions = list(product([0, 1], repeat=10))

    best_solution = None
    best_benefit = 0
    for solution in all_solutions:
        benefit_i, volume_i = evaluate_individual(solution)
        if volume_i <= CAPACITY:
            if benefit_i > best_benefit:
                best_benefit, best_solution = benefit_i, solution

    print("Best individual: ", best_solution)
    best_benefit, best_volume = evaluate_individual(best_solution)
    print("Benefit: ", best_benefit)
    print("Volume: ", best_volume)

    print("Brute force took", str(time.time() - start_time)[0:6], "seconds to run")


def hill_climber(EPOCH=100):
    start_time = time.time()

    individual = random_individual()
    scores = []
    for generation in range(EPOCH):

        old_fitness = fitness(individual)
        new_individual = mutate(individual)
        new_fitness = fitness(new_individual)

        if new_fitness > old_fitness:
            individual = new_individual
            scores.append(new_fitness)
        else:
            scores.append(old_fitness)

    print("Best gene: ", individual)
    benefit, volume = evaluate_individual(individual)
    print("Benefit: ", benefit)
    print("Capactiy: ", volume)

    print("Hill climber took", str(time.time() - start_time)[0:6], "seconds to run")

    return scores


# ### Population of Hill Climbers
def flip(individual, index):
    individual = deepcopy(individual)
    if individual[index] == 0:
        individual[index] = 1
    else:
        individual[index] = 0
    return individual

def mutate_matrix(matrix):
    matrix = deepcopy(matrix)
    pop_size = matrix.shape[0]

    random_indices = np.random.randint(low = 0, high = NUM_GENES, size = pop_size)
    mutated_matrix = np.zeros(matrix.shape)

    for i in range(len(matrix)):
        old_individual = matrix[i,:]
        new_individual = flip(old_individual, random_indices[i])
        old_score = fitness(old_individual)
        new_score = fitness(new_individual)

        if old_score > new_score:
            mutated_matrix[i,:] = old_individual
        else:
            mutated_matrix[i,:] = new_individual

    return mutated_matrix

def matrix_to_scores(matrix):
    matrix = deepcopy(matrix)
    pop_size = matrix.shape[0]
    scores = np.zeros(pop_size)
    for i in range(pop_size):
        scores[i] = fitness(matrix[i,:])
    return scores

def hill_climbers(population_size=200, EPOCH=50000):

    scores = np.zeros((population_size, EPOCH))
    population = np.array(random_population(population_size))

    for epoch in range(EPOCH):
        population = mutate_matrix(population)
        scores_epoch = matrix_to_scores(population)
        scores[:, epoch] = scores_epoch

    benefits = []
    for individual in range(population_size):
        benefit, volume = evaluate_individual(population[individual,:])
        benefits.append(("benefit:", benefit, " volume:", volume, "score:", scores[individual, EPOCH-1]))

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Hill Climbers")
    plt.plot(np.transpose(scores));

    return benefits



# Every film should have its own world, a logic and feel to it that expands beyond the exact image that the audience is seeing.
# When I make a film, I am hoping to reinvent the genre a little bit. I just do it my way. I make my own little Quentin versions of them...
