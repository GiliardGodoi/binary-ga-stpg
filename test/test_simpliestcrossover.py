import os
import random
import unittest
from pprint import pprint

from graph import Graph, ReaderORLibrary
from ga_simplestpartition import PXSimpliestGeneticAlgorithm

from graph.steiner import shortest_path_with_origin
from graph.util import has_cycle, is_steiner_tree
from genetic.chromosome import TreeBasedChromosome
from tools import evaluate_treegraph
from genetic.datalogger import BaseLogger
from px_simpliest import SimplePartitionCrossover


class TestSimpliestPartitionCrossover(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        filename = os.path.join("datasets", "ORLibrary", "steinb13.txt")
        reader = ReaderORLibrary()
        self.STPG = reader.parser(filename)
        self.GRAPH = Graph(edges=self.STPG.graph)
        self.SPX = SimplePartitionCrossover(graphinstance=self.GRAPH)

        random.seed(9876543210)

    def test_dataset(self):
        '''Leitura da instância do problema'''

        self.assertEqual(self.STPG.name, "B13")
        self.assertEqual(self.STPG.nro_nodes, 100)
        self.assertEqual(self.STPG.nro_edges, 125)
        self.assertEqual(self.STPG.nro_terminals, 17)


    def test_crossover(self):
        '''Operador de cruzamento está executando?

        Avalia tão somente se o operador de cruzamento está executando e se o resultado é uma árvore de steiner.
        '''

        s1, s2 = random.sample(self.STPG.terminals, k=2)
        SUBTREE_A, cost1 = shortest_path_with_origin(self.GRAPH, s1, self.STPG.terminals)
        SUBTREE_B, cost2 = shortest_path_with_origin(self.GRAPH, s2, self.STPG.terminals)
        offspring = self.SPX.operation(TreeBasedChromosome(SUBTREE_A), TreeBasedChromosome(SUBTREE_B))

        is_smt, result = is_steiner_tree(offspring.graph, self.STPG)
        if not is_smt:
            pprint(result)
        self.assertTrue(is_smt)
        self.assertFalse(has_cycle(offspring.graph))

    def test_run_ga(self):

        STPG = self.STPG

        GA = PXSimpliestGeneticAlgorithm(STPG)
        GA.logger = BaseLogger()

        GA.generate_population(population_size=10)
        MAX_GENERATION = 100
        counter = 0

        while counter < MAX_GENERATION:
            # print("Iteration: ", counter + 1, end='\r')
            GA.evaluate(iteration=counter)
            GA.sort_population()
            GA.selection()
            GA.recombine()
            counter += 1

        GA.evaluate(iteration=counter+1)

