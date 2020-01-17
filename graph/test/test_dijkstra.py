import unittest
from os import path

from graph import GraphDictionary as Graph
from graph.reader import Reader
from graph.algorithms import shortest_path_dijkstra as dijkstra

class TestDijkstraImplementation(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        diretorio_dados = "datasets"
        arquivo_dados = "b01.stp"
        arquivo = path.join(diretorio_dados, arquivo_dados)

        reader = Reader()

        self.stp = reader.parser(arquivo)

        self.graph = Graph(vertices=self.stp.nro_nodes,edges=self.stp.graph)

    def test_distance(self):
        start_node = 15
        dist, _ = dijkstra(self.graph, start_node)

        self.assertEqual(dist[start_node],0)

if __name__ == "__main__" :
    unittest.main()