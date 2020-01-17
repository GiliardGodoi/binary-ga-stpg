# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:35:51 2020

@author: Giliard A Godoi
"""
from os import path
from collections import deque

from graph import GraphDictionary, Reader
from graph.steiner_heuristics import shortest_path_origin_prim
from util import gg_union


search_tree = set()
return_branch = set()

def visitar(v,u, main_branch : bool):

    min_max_edge = lambda x, y : (min(x,y), max(x,y))

    if main_branch:
        search_tree.add(min_max_edge(v,u))
    else:
        return_branch.add(min_max_edge(v,u))


def busca_produndidade(grafo : GraphDictionary, start_node):

    vertices_done = set()
    stack = deque()

    def P(v):
        vertices_done.add(v) # vertice marcado
        stack.append(v) # vertice em stack

        for w in grafo.adjacent_to(v):
            if not w in vertices_done:
                visitar(v, w, True)
                P(w)
            elif (w in stack) and w != stack[-2]:
                visitar(v,w, False)

        stack.pop()

    P(start_node)


if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = GraphDictionary(vertices=stp.nro_nodes, edges=stp.graph)

    sub1, c1 = shortest_path_origin_prim(graph, stp.terminals[5], stp.terminals) # 0 ate 16
    sub2, c2 = shortest_path_origin_prim(graph, stp.terminals[16], stp.terminals)

    GU = gg_union(sub1, sub2)

    busca_produndidade(GU, 19)