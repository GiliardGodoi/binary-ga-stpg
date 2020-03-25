# -*- coding: utf-8 -*-
import random

## SELECTIONS METHODS

def tournament_selection(population):
    selected = list()
    pool_size = len(population)
    count = 0

    while count < pool_size:
        c1, c2 = random.sample(population, k=2)
        if c1.fitness < c2.fitness:
            selected.append(c1)
        else:
            selected.append(c2)

        count += 1

    return selected


def roullete_selection(population):
    pool_size = len(population)
    fitnesses = [c.fitness for c in population ]

    # Return a k sized list of population elements chosen with replacement
    selected = random.choices(population, weights=fitnesses, k=pool_size)

    return selected

##  CROSSOVER METHODS FOR BINARY CHROMOSOMES

def crossover_2points(A_parent, B_parent):
    length = len(A_parent)
    points = random.sample(range(0, length), k=2)
    points.sort()
    p1, p2 = points

    crossing = lambda genex, geney : genex[:p1] + geney[p1:p2] + genex[p2:]

    offspring_A = Chromosome(crossing(A_parent.genes, B_parent.genes))
    offspring_B = Chromosome(crossing(B_parent.genes, A_parent.genes))

    return offspring_A, offspring_B


def crossover_1points(A_parent, B_parent):
    length = len(A_parent)
    point = random.choice(range(0,length))

    crossing = lambda genex, geney : genex[:point] + geney[point:]

    offspring_A = Chromosome(crossing(A_parent.genes, B_parent.genes))
    offspring_B = Chromosome(crossing(B_parent.genes, A_parent.genes))

    return offspring_A, offspring_B

## MUTATION METHODS

def mutation_flipbit(chromosome):
    '''Flip exactly one bit from the chromosome genes'''

    flipbit = lambda x : '1' if x == '0' else '0'

    index = random.randrange(0, len(chromosome))
    genes = chromosome.genes
    chromosome.genes = genes[:index] + flipbit(genes[index]) + genes[(index + 1):]

    return chromosome


## NORMALIZATION METHODS


## CHROMOSOME CLASS

class Chromosome(object):

    def __init__(self, genes, fitness=0):

        self.genes = genes
        self.__fitness = 0
        self.__score = fitness
        self.normalized = False

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value
        self.__score = value
        self.normalized = False

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, value):
        self.__score = value
        self.normalized = True

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return str(self.genes)

    def __repr__(self):
        return self.__str__()


class GeneticAlgorithm(object):
    '''
    Basic implementation of mainly fuctions for a GA
    with binary chromosome representation.
    '''
    def __init__(self, graph, terminals):
        self.graph = graph
        self.terminals = set(terminals)

        self.population = list()
        self.popultaion_size = 0

        self.best_chromossome = None
        self.chromosome_length = len(graph) - len(terminals) # quantidade de vertices no grafo

        self.tx_mutation = 0.2
        self.tx_crossover = 0.85

    def fitness(self, chromosome):
        return 0

    def generate_individual(self):
        length = self.chromosome_length

        genes = ''.join(random.choices(['0', '1'], k=length))

        return Chromosome(genes)

    def generate_population(self, population_size):

        assert population_size > 0, "Population must have more than 0 individuals"
        assert (population_size % 2) == 0, "Population size must be a even number"
        self.population_size = population_size

        population = list()

        for _ in range(0, population_size):
            chromosome = self.generate_individual()
            population.append(chromosome)

    def evaluate(self):
        population_fitness = 0

        for chromosome in self.population:
            fit = self.fitness(chromosome)
            chromosome.fitness = fit
            population_fitness += fit

        return population_fitness

    def selection(self):
        self.population = roullete_selection(self.population)

    def recombine(self):

        newpopulation = list()
        population_size = self.population_size
        count = 0

        while count < population_size:
            parent_a, parent_b = random.sample(self.population, k=2)
            child_a, child_b = crossover_2points(parent_a, parent_b)

            if child_a.fitness < child_b.fitness :
                newpopulation.append(child_a)
            else :
                newpopulation.append(child_b)
            count += 1

        self.updata_population(newpopulation)

    def mutation(self):
        population_size = self.population_size
        count = 0

        while count < population_size:
            if random.random() < self.tx_mutation:
                self.population[count] = mutation_flipbit(self.population[count])
            count += 1

    def elitism(self):
        pass

    def updata_population(self, new_population):
        '''It's implements the population replace strategy'''

        assert len(self.population) == len(new_population), "It is not the same size"
        ## Replace all
        self.population = new_population

    def normalize(self):
        pass
