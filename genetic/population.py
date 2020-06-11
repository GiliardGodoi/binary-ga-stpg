
import statistics
from random import choices

from genetic.datalogger import BaseLogger
from genetic.individual import Individual

class BasePopulation:
    '''Define the basic class to define a GA'''

    def __init__(self, chromosomes,
                 evaluation_func,
                 intended_size = None,**kwargs):

        self.generation = 0
        self.individuals = [ Individual(chromosome) for chromosome in chromosomes]
        self.intended_size = intended_size or len(self.individuals)

        self.evaluation_func = evaluation_func
        self.best_chromosome = None
        self.datacolector = BaseLogger()

    def __getitem__(self, i):
        return self.individuals[i]

    def __len__(self):
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    @property
    def current_best(self):
        evaluated_individuals = tuple(filter(lambda x: x.fitness is not None, self.individuals))
        if len(evaluated_individuals) > 0:
            return max(evaluated_individuals, key=getattr("fitness"))

    @property
    def is_evaluated(self):
        return all(individual.cost is not None for individual in self.individuals)

    @property
    def is_normalized(self):
        return all(individuo.is_normalized for individuo in self.individuals)

    def evaluate(self, **kwargs):
        '''Evaluates the entire population.

        Returns:
            int or float : total population cost
            int or float : maximun cost from the current generation
            float : average cost from the current generation
        '''
        evaluation_func = self.evaluation_func
        evaluated_costs = list()
        count_penalized = 0
        for individuo in self.individuals:
            cost, partitions = individuo.evaluate(evaluation_func)
            evaluated_costs.append(cost)
            if partitions > 1: count_penalized += 1

    def select(self, selector_func, **kwargs):
        self.selected_population = selector_func(self.individuals, **kwargs)

        return self

    def recombine(self, mate_func, **kwargs):

        newpopulation = list()
        intended_size = self.intended_size

        while len(newpopulation) < intended_size:
            parent_a, parent_b = choices(self.selected_population, k=2)
            result = mate_func(parent_a.chromosome, parent_b.choromose, **kwargs)
            if isinstance(result, (list, tuple)) :
                newpopulation.extend(Individual(child) for child in result)
            else:
                newpopulation.append(Individual(result))

        self._update_population(newpopulation)

    def mutation(self, mutate_func,tx_mutation=0.2, **kwargs):
        for individuo in self.individuals:
                individuo.mutate(mutate_func=mutate_func,probability=tx_mutation, **kwargs)

        return self

    def normalize(self, **kwargs):

        current_best = None

        max_cost = max(chromosome.cost for chromosome in self.individuals)
        population_fitness = list()

        for chromosome in self.individuals:
            fitness = max_cost - chromosome.cost
            chromosome.fitness = fitness
            population_fitness.append(fitness)

            if ((current_best is None) or
                chromosome.fitness > current_best.fitness):
                current_best = chromosome

        self.datacolector.log("evaluation",
            self.generation,
            kwargs.get("penalized", None),
            statistics.mean(population_fitness),
            statistics.stdev(population_fitness))

        self._update_best_chromosome(current_best, **kwargs)

        return self

    def survivor(self, **kwargs):
        return self

    def _update_population(self, newpopulation, **kwargs):
        '''It's execute the population replace strategy'''

        assert len(newpopulation) == self.intended_size, "It is not the same size"
        ## Replace all
        self.population = newpopulation

    def _update_best_chromosome(self, current_best, **kwargs):
        if (self.best_chromosome is None  or
            self.best_chromosome.cost > current_best.cost):
            self.best_chromosome = current_best
            self.datacolector.log('best_fitness', kwargs.get("iteration", 0), current_best.cost, current_best.fitness)
            self.best_chromosome.last_improvement = self.generation

        self.datacolector.log('best_from_round', kwargs.get("iteration", 0), current_best.cost, current_best.fitness)

    def sort_population(self,key=attrgetter("fitness")):
        '''Sort the population by fitness attribute'''
        self.individuals.sort(key=key)
        return self

    def map(self, func, **kwargs):
        self.individuals = [func(individuo, **kwargs) for individuo in self.individuals]
        return self

    def filter(self, func, **kwargs):
        self.individuals = [ individuo for individuo in self.individuals if func(individuo, **kwargs)]
        return self

    def callback(self, func, **kwargs):
        func(self.individuals, **kwargs)
        return self
