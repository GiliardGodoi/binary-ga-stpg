
from random import random
import reprlib

class Individual:

    def __init__(self, chromosome, cost=None, **kwargs):
        self.chromosome = chromosome
        self.age = 0
        self.last_improvement = 0
        self._cost = cost
        self._fitness = cost
        self.is_normalized = False
        self.partitions = 0

    def __repr__(self):
        return f"{self.__class__.__name__}({reprlib.repr(self.chromosome)}, cost : {self.cost})"

    def __copy__(self):
        newer = Individual(chromosome=self.chromosome, cost=self.cost)
        newer.fitness = self.fitness
        newer.is_normalized = self.is_normalized
        newer.partitions = self.partitions
        newer.age = self.age
        newer.last_improvement = self.last_improvement

        return newer

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value
        self._fitness = None
        self.is_normalized = False

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        if value is not None:
            self._fitness = value
            self.is_normalized = True

    def evaluate(self, eval_func, **kwargs):
        result = eval_func(self.chromosome)

        if isinstance(result, (int, float)):
            self.cost, self.partitions = result, None
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            self.cost, self.partitions = result
        else:
            raise RuntimeError(f"Could not interpret return {result}")

        return self.cost, self.partitions

    def mutate(self, mutate_func, probability, **kwargs):

        if random() < probability:
            self.chromosome = mutate_func(self.chromosome, **kwargs)
            self.cost = None