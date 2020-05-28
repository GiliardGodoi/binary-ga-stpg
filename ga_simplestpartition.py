# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:26:16 2020

@author: Giliard Almeida de Godoi
"""
from genetic.base import BaseGA
from genetic.chromosome import TreeBasedChromosome
from genetic.datalogger import DataLogger
from genetic.selection import roullete_selection
from graph import Graph, SteinerTreeProblem
from tools import evaluate_treegraph, random_treegraph_chromosome
from px_simpliest import SimplePartitionCrossover

class PXSimpliestGeneticAlgorithm(BaseGA):

    def __init__(self, STPG : SteinerTreeProblem, parameters):
        super().__init__(parameters)

        self.STPG = STPG
        self.GRAPH = Graph(edges=STPG.graph)
        self.terminals = set(STPG.terminals)

        self.crossover_operator = SimplePartitionCrossover(self.GRAPH)
        self.selection_operator = roullete_selection

        self.logger = DataLogger()

    def generate_new_individual(self, **kwargs):
        return random_treegraph_chromosome(self.GRAPH, self.terminals)

    def eval_chromosome(self, chromosome : TreeBasedChromosome):

        def penality(qtd_partition):
            return (qtd_partition - 1) * 100

        return evaluate_treegraph(chromosome, penality)

    def mutation(self):
        pass
