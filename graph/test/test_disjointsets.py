import unittest

from graph.graph import Graph
from graph.util import has_cycle, gg_total_weight, gg_edges_number
from graph.algorithms import kruskal

class TestDisjointSets(unittest.TestCase):

    def setUp(self):

        self.tree = Graph()

        self.tree.add_edge('a', 'b')
        self.tree.add_edge('a', 'c')
        self.tree.add_edge('a', 'd')
        self.tree.add_edge('b', 'e')
        self.tree.add_edge('b', 'f')
        self.tree.add_edge('c', 'g')
        self.tree.add_edge('d', 'h')
        self.tree.add_edge('d', 'i')
        self.tree.add_edge('d', 'j')
        self.tree.add_edge('j', 'k')
        self.tree.add_edge('j', 'l')

    def test_no_cycle(self):

        self.assertFalse(has_cycle(self.tree))

    def test_has_cycle(self):

        self.tree.add_edge('g', 'a') ## has a cycle
        self.assertTrue(has_cycle(self.tree))

    def teste_kruskal_mst_algorithm(self):
        '''Kruskal's algorithm uses Disjoint Set under the hook'''
        edges = [
                ('a', 'b', 4), ('a', 'h', 8), ('h', 'b', 11),
                ('h', 'i', 7), ('h', 'g', 1), ('i', 'g', 6),
                ('i', 'c', 2), ('b', 'c', 8),
                ('c', 'f', 4), ('g', 'f', 2),
                ('d', 'f', 14), ('c', 'd', 7),
                ('d', 'e', 9), ('e', 'f', 10),
            ]
        graph = Graph()
        for v, u, w in edges:
            graph.add_edge(v, u, weight=w)

        tree = kruskal(graph)

        nro_edges = gg_edges_number(tree)
        cost = gg_total_weight(tree)

        self.assertEqual(nro_edges, 8)
        self.assertEqual(cost, 37)


if __name__ == "__main__":
    unittest.main()