from graph import Reader
from graph import GraphDictionary
# from graph.util.path_finder import find_path
from graph.util.path_finder import find_shortest_path

from os import path

diretorio_dados = "datasets"
arquivo_dados = "b01.stp"

arquivo = path.join(diretorio_dados, arquivo_dados)

reader = Reader()

stp = reader.parser(arquivo)

import pprint as pp 

pp.pprint(stp.graph)

graph = GraphDictionary(vertices=stp.nro_nodes, terminals=stp.terminals, edges=stp.graph)

path = find_shortest_path(graph,1, 37)