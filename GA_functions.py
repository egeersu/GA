import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Global variables (specific to the problem at hand)
# TODO: make these modifiable by the user (A GA_Problem class?)
items = np.array([(5,3), (6,2), (1,4), (9,5), (2,8), (8,9), (4,10), (3,1), (7,6), (10,7)])
CAPACITY = 20
NUM_GENES = len(items)

# #### Random Individual
# * Generates a random individual with `length=10` number of genes.
def random_individual(length=NUM_GENES):
    return np.random.randint(low = 0, high = 2, size = length)

# #### Random Population
# * Generates a population with `population_size` number of individuals.
# * Each individual will have `length` number of genes.
def random_population(population_size=100, length=10, vectorized=False):
    list_of_individuals = [random_individual(length) for i in range(population_size)]
    if vectorized:
        return np.array(list_of_individuals)
    else:
        return list_of_individuals

# #### Evaluate Individual
# * Given an individual, returns the total benefit and total volume.
def evaluate_individual(individual):
    total_benefit = sum(individual * items[:, 0])
    total_volume = sum(individual * items[:, 1])
    return total_benefit, total_volume

# #### Mutate
# * Given an individual, flip a random gene.
def mutate(individual):
    individual = deepcopy(individual)
    index = random.randint(0,len(individual)-1)
    if individual[index] == 0:
        individual[index] = 1
    else:
        individual[index] = 0
    return individual

# #### Mutate Population
# * Given a population, mutate all individuals. Return a new population.
# * Don't modify the original population.
def mutate_population(population):
    new_population = []
    for individual in population:
        new_population.append(mutate(individual))
    return new_population

# #### Fitness
# * Given an individual, evaluate its fitness.
def fitness(individual):
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
    best_solution = None
    best_fitness = 0
    for i in range(epochs):
        indiv_i = random_individual()
        fitness_i = fitness(indiv_i)
        if fitness_i > best_fitness:
            best_fitness, best_solution = fitness_i, indiv_i

    print("Best individual: ", best_solution)
    best_benefit, best_volume = evaluate_individual(best_solution)
    print("Benefit: ", best_benefit)
    print("Volume: ", best_volume)


### Brute Force
# The best solution turns out to have:
# * **Benefit**: 33
# * **Volume**: 18
def brute_force():
    # enumerate all possible solutions
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

def hill_climber(EPOCH=30):
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
    return scores

def hill_climber_graph():
    scores = hill_climber()
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Single Hill Climber")
    plt.plot(scores);


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
