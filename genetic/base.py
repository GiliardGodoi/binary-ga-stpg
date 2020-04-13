import random
from operator import attrgetter

from genetic.chromosome import BaseChromosome as Chromosome
from genetic.crossover import crossover_2points
from genetic.mutation import mutation_flipbit
from genetic.selection import roullete_selection


class BaseGA:
    '''Define the basic class to define a GA'''

    def __init__(self, **kargs):

        self.population = list()
        self.population_size = 0

        self.tx_crossover = 0.9
        self.tx_mutation = 0.2

        self.chromosome_legth = 10

        self.logger = None

    @staticmethod
    def build_parametrized_ga(**kargs):
        pass

    def eval_chromosome(self, chromosome : Chromosome):
        raise NotImplementedError("")

    def generate_population(self):
        raise NotImplementedError("")

    def evaluate(self, **kargs):
        for chromosome in self.population:
            chromosome.cost = self.eval_chromosome(chromosome)

    def selection(self):
        raise NotImplementedError("")
        # self.population = self.operator_strategy.apply(self.population, self.population_size)

    def recombine(self):
        raise NotImplementedError("")
        # newpopulation = list()
        # population_size = self.population_size
        # count = 0

        # while count < population_size:
        #     parent_a, parent_b = random.sample(self.population, k=2)
        #     child_a, child_b = self.operator_crossover.apply(parent_a, parent_b)

        #     newpopulation.append(child_a)
        #     newpopulation.append(child_b)
        #     count += 2

        # self.updata_population(newpopulation)

    def mutation(self):
        population_size = self.population_size
        count = 0

        while count < population_size:
            if random.random() < self.tx_mutation:
                self.population[count] = self.operator_mutation.apply(self.population[count])
            count += 1

    def normalize(self):
        raise NotImplementedError("")

    def update_population(self, newpopulation):
        raise NotImplementedError("")

    def update_best_chromosome(self, chromosome):
        raise NotImplementedError("")

    def sort_population(self):
        '''Sort the population by fitness attribute'''
        self.population.sort(key=attrgetter("fitness"))


class Operator:

    def __init__(self):
        pass

    def __call__(self, *args, **kargs):
        return self.operation(*args, **kargs)

    def operation(self, *args, **kargs):
        raise NotImplementedError("This method must be implemented by the subclass")
