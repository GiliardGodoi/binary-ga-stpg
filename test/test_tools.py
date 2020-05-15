import os
import random
import unittest

from ga_simplestpartition import SimplePartitionCrossover
from genetic.chromosome import BinaryChromosome, TreeBasedChromosome
from graph import Graph, ReaderORLibrary
from graph.algorithms import prim
from graph.steiner import shortest_path_with_origin
from graph.util import has_cycle, is_steiner_tree
from util import (convert_binary2treegraph,
                convert_treegraph2binary,
                evaluate_binary,
                evaluate_treegraph,
                random_binary_chromosome,
                random_treegraph_chromosome,
                vertices_from_binary_chromosome)


class TestToolFunctions(unittest.TestCase):
    '''Teste dos métodos auxiliares para o módulo util.py'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        filename = os.path.join("datasets", "ORLibrary", "steinb13.txt")
        reader = ReaderORLibrary()
        self.STPG = reader.parser(filename)
        self.GRAPH = Graph(edges=self.STPG.graph)
        self.SPX = SimplePartitionCrossover(graphinstance=self.GRAPH)

        random.seed(789456123)
        # random.seed(123456789)

    def test_dataset(self):
        '''Leitura da instância do problema'''

        self.assertEqual(self.STPG.name, "B13")
        self.assertEqual(self.STPG.nro_nodes, 100)
        self.assertEqual(self.STPG.nro_edges, 125)
        self.assertEqual(self.STPG.nro_terminals, 17)

    def test_convert_binary2tree(self):
        length = self.STPG.nro_nodes - self.STPG.nro_terminals
        binary = random_binary_chromosome(length)

        vertices = vertices_from_binary_chromosome(binary,
                                                self.STPG.terminals,
                                                self.STPG.nro_nodes)

        subtree = convert_binary2treegraph(binary,
                                        self.GRAPH,
                                        self.STPG.terminals,
                                        self.STPG.nro_nodes)

        costbinary, _ = evaluate_binary(binary,
                        self.GRAPH,
                        self.STPG.terminals,
                        self.STPG.nro_nodes,
                        lambda k: (k-1) * 100)

        costtree, _ = evaluate_treegraph(subtree,
                                    lambda k : (k-1) * 100)

        self.assertSetEqual(vertices, set(subtree.graph.vertices))
        self.assertEqual(costbinary, costtree)


    def test_convert_tree2binary(self):
        '''Converte a representação em árvore para a representação binária.
        '''
        terminal = 58
        subtree, cost1 = shortest_path_with_origin(self.GRAPH, terminal, self.STPG.terminals)

        chromosome = convert_treegraph2binary(TreeBasedChromosome(subtree),
                                                self.STPG.terminals,
                                                self.STPG.nro_nodes)

        chromosome_length = self.STPG.nro_nodes - self.STPG.nro_terminals
        self.assertEqual(len(chromosome.genes), chromosome_length)

        vertices = vertices_from_binary_chromosome(chromosome,
                                            self.STPG.terminals,
                                            self.STPG.nro_nodes)

        self.assertSetEqual(vertices, set(subtree.vertices))

    def test_compare_evaluatebinary_primmst(self):
        '''Função de aptidão para cromossomo com representação binária.

        Define um cromossomo com todos os vértices não-obrigatórios como vértices de steiner.
        Todos estão presentes na solução parcial.

        A função de aptidão deverá retornar o custo da MST da árvore.

        Utiliza o algoritmo de Prim para cálcular a MST do grafo que serve de instância para o problema.

        O custo da MST cálculada pelos dois procedimentos distintos deverá ser igual.
        '''
        length = self.STPG.nro_nodes - self.STPG.nro_terminals

        chromosome = BinaryChromosome(''.join(['1'] * length))
        tree, cost1 = prim(self.GRAPH, random.choice(self.STPG.terminals))

        cost2, is_disconnected = evaluate_binary(chromosome,
                                                self.GRAPH,
                                                self.STPG.terminals,
                                                self.STPG.nro_nodes,
                                                lambda k: (k-1) * 100)

        self.assertEqual(cost1, cost2)
        self.assertFalse(is_disconnected)

    def test_evaluate_tree(self):
        '''Função de aptidão para o cromossomo com representação em árvore

        A função de avaliação do cromossomo retornao mesmo custo incialmente calculado para uma
        solução candidata obtida por meio de uma heurística.
        '''

        s1 = random.choice(self.STPG.terminals)
        SUBTREE_A, cost1 = shortest_path_with_origin(self.GRAPH, s1, self.STPG.terminals)
        result, _ = is_steiner_tree(SUBTREE_A, self.STPG)
        cost2, k = evaluate_treegraph(TreeBasedChromosome(SUBTREE_A), lambda k : (k-1) * 100 )

        self.assertTrue(result)
        self.assertEqual(cost1, cost2)
        self.assertEqual(k, 0)

if __name__ == "__main__":
    unittest.main()
