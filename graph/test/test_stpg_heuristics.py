import random
import unittest
from collections import deque
from os import path

from graph.graph import Graph
from graph.reader import Reader
from graph.steiner import (prunning_mst, shortest_path,
                           shortest_path_origin_prim,
                           shortest_path_with_origin)
from graph.util import has_cycle


class TestSTPGHeuristicas(unittest.TestCase):

    def setUp(self):
        reader = Reader()
        self.stpg_instance = reader.parser(path.join("datasets", "osti", "b13.stp"))

        self.graph = Graph(vertices = self.stpg_instance.nro_nodes,
                            edges=self.stpg_instance.graph)

        random.seed(1234567890)


    def test_instance_reading(self):

        stpg = self.stpg_instance

        self.assertEqual(stpg.nro_edges, 125)
        self.assertEqual(stpg.nro_nodes, 100)
        self.assertEqual(stpg.nro_terminals, 17)

        self.assertEqual(stpg.nro_terminals, len(stpg.terminals))
        self.assertEqual(stpg.nro_nodes, len(stpg.graph))

    def test_shortest_path(self):
        graph = self.graph
        stpg = self.stpg_instance

        terminal = random.choice(stpg.terminals)
        gg, cost = shortest_path(graph, terminal, stpg.terminals)

        self.common_cases(gg, cost)

    def test_shortest_path_with_origin(self):

        graph = self.graph
        stpg = self.stpg_instance

        terminal = random.choice(stpg.terminals)
        gg, cost = shortest_path_with_origin(graph, terminal, stpg.terminals)

        self.common_cases(gg, cost)

    def test_shortest_path_origin_prim(self):

        graph = self.graph
        stpg = self.stpg_instance

        terminal = random.choice(stpg.terminals)
        gg, cost = shortest_path_origin_prim(graph, terminal, stpg.terminals)

        self.common_cases(gg, cost)

    def test_prunning_mst(self):

        graph = self.graph
        stpg = self.stpg_instance

        terminal = random.choice(stpg.terminals)
        gg, cost = prunning_mst(graph, terminal, stpg.terminals)

        self.common_cases(gg, cost)


    def common_cases(self, steiner_tree : Graph, cost : int):

        self.assertIsInstance(steiner_tree, Graph)
        self.assertIsInstance(cost, int)
        self.assertGreater(cost,0)

        terminals = set(self.stpg_instance.terminals)

        ## Se o vértice possui grau 1 então é terminal. Mas se for terminal possui grau 1?
        degrees = { k : len(steiner_tree[k]) for k in steiner_tree.edges.keys() }
        for k, v in degrees.items() :
            if v == 1 :
                is_terminal = (k in terminals)
                self.assertTrue(is_terminal)

        all_vertices = set(steiner_tree.vertices)

        self.assertIsInstance(all_vertices, set)
        self.assertEqual(len(all_vertices), len(steiner_tree.vertices))

        ## todos os vertices terminais estao contidos na solução
        tt = terminals - all_vertices
        self.assertFalse(tt)

        ## Existe algum ponto de steiner na solução. Mas sempre isso vai acontecer?
        ss = all_vertices - terminals
        self.assertTrue(ss)

        cycles = has_cycle(steiner_tree)
        self.assertFalse(cycles)

if __name__ == "__main__" :
    unittest.main()
