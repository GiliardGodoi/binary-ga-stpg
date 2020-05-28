import os
import random
import statistics
import time
from collections import deque

from px_simpliest import SimplePartitionCrossover
from genetic.base import BaseGA
from genetic.chromosome import BinaryChromosome, TreeBasedChromosome
from genetic.crossover import crossover_2points
from graph import Graph, SteinerTreeProblem
from tools import (convert_binary2treegraph, convert_treegraph2binary,
                  evaluate_binary, evaluate_treegraph,
                  random_binary_chromosome, random_treegraph_chromosome)

class HybridGeneticAlgorithm(BaseGA):

    def __init__(self, STPG : SteinerTreeProblem, parameters):
        super().__init__(parameters)

        self.STPG = STPG
        self.GRAPH = Graph(edges=STPG.graph)
        self.terminals = set(STPG.terminals)
        self.nro_vertices = STPG.nro_nodes
        self.chromosome_length = STPG.nro_nodes - STPG.nro_terminals

        self.CONTROL_FLAG = True # if True --> BinaryRepresentation
        self.gpx_strategy = SimplePartitionCrossover(self.GRAPH)

    def generate_new_individual(self, **kwargs):
        return random_binary_chromosome(self.chromosome_length)

    def eval_chromosome(self, chromosome):

        def penality(nro_partitions):
            return (nro_partitions - 1) * 100

        if type(chromosome) is BinaryChromosome:
            return evaluate_binary(chromosome,self.GRAPH, self.terminals, self.nro_vertices, penality)

        elif type(chromosome) is TreeBasedChromosome:
            return evaluate_treegraph(chromosome, penality)
        else:
            raise TypeError("chromosome cannot be evaluated")

    def mutation(self):
        if self.CONTROL_FLAG:
            super().mutation()

    def check_it(self, **kwargs):
        self.last_time_improvement += 1

        change_interval = self.change_interval

        iteration = kwargs.get("iteration", None)
        if (iteration % change_interval) == 0:
            self.change_representation()

    def change_representation(self):
        print("changing ...                                         ")

        # 1st thing to do: change the control flag
        self.CONTROL_FLAG = not self.CONTROL_FLAG

        # 2nd one is change the crossover strategy
        self.crossover_operator = crossover_2points if self.CONTROL_FLAG else self.gpx_strategy
        print(self.crossover_operator.__class__.__name__) ##

        # 3rd one is convert the chromosome representation
        newpopulation = list()
        for chromosome in self.population:
            if self.CONTROL_FLAG:
                newpopulation.append(convert_treegraph2binary(chromosome, self.terminals, self.nro_vertices))
            else:
                newpopulation.append(convert_binary2treegraph(chromosome,self.GRAPH, self.terminals, self.nro_vertices))

        # 4th one is update the population
        self.population = newpopulation
