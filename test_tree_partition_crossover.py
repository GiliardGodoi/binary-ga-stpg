# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2020

@author: Giliard A Godoi
"""
from os import path

from graph import GraphDictionary, Reader
from graph.steiner_heuristics import shortest_path_with_origin
from util import gg_rooted_tree

class Subset:
    
    def __init__(self,parent, rank):
        self.parent = parent
        self.rank = rank
        

def find(subsets : list):
    pass


def union(subsets,u : Subset,v : Subset):
    pass


if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = GraphDictionary(vertices=stp.nro_nodes, edges=stp.graph)

    s1 = stp.terminals[5]
    subtree_1, cost1 = shortest_path_with_origin(graph, s1, stp.terminals) # 0 ate 16

    s2 = stp.terminals[9]
    subtree_2, cost2 = shortest_path_with_origin(graph, s2, stp.terminals)

    print(cost1, cost2)

    # Primeiro uma das soluções será transformada na sua forma enraizada
    rooted_tree_1 = gg_rooted_tree(subtree_1,s1)

    # Definir o conjunto de arestas que somente pertence a subtree_2
    edges_disjoint_2 = set()

    # uma função auxiliar para padronizar as arestas. Será necessário?
    std_edges = lambda x, y : (min(x,y), max(x,y))
    
    
    # todas as arestas não comum de um dos pais (subtree 2)
    for v, u in subtree_2.gen_undirect_edges():
        if not subtree_1.has_edge(v,u) :
            edges_disjoint_2.add((v,u))
            
    

