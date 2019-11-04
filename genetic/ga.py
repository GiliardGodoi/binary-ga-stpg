# -*- coding: utf-8 -*-
import random
from genetic.chromosome import Chromosome

class GeneticAlgorithm(object):
    '''
        This class provide a backbone for the GA's stepwise.
        The implementation for each step should remain in separate modules.
    '''

    def __init__(self,population_size,max_generation):
        # meta-data
        self.population_size = population_size
        self.max_generation = max_generation

        # population
        self.population = None

    def run(self):
        
        satisfy_stop_criteria = self.define_stop_criteria(MAX_GENERATION = self.max_generation)

        self.inicialize_population()
        self.evaluate_population()

        i = 0
        while not satisfy_stop_criteria(i) :
            
            self.normalize_population()
            new_population = list()
            for A, B in self.selection():
                child = self.crossover(A,B)
                child = self.mutation(child)
                new_population.append(child)

            self.elitism(new_population)
            self.updata_population(new_population)
            self.evaluate_population()
            i += 1

    def define_stop_criteria(self,**params):
        if "MAX_GENERATION" in params:

            max_gen = params["MAX_GENERATION"]
            return lambda i : max_gen <= i 

        raise RuntimeError()

    def inicialize_population(self):
        pass

    def evaluate_population(self):
        pass

    def updata_population(self,new_population):
        pass

    def normalize_population(self):
        pass

    def selection(self):
        for _ in range(0,50):
            a = [ random.randint(5,10) for _ in range(10) ]
            b = [ random.randint(0,4) for _ in range(10) ]
            yield (a,b)

    def crossover(self,parentA, parentB):
        child = list()
        for a, b in zip(parentA, parentB) :
            if random.randint(0,1) :
                child.append(a)
            else:
                child.append(b)

        return child

    def mutation(self, child):
        if random.random() > 0.8:
            x = random.randint(0,len(child)-1)
            child[x] = random.randint(0,10)

        return child

    def elitism(self, new_population):
        pass


    def collect_generation_logdata(self,**params):
        pass
