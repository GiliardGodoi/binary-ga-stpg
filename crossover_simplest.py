# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:26:16 2020

@author: Giliard Almeida de Godoi
"""
from os import path
import random
# from collections import deque

from graph import Graph, Reader
from graph.steiner_heuristics import shortest_path_with_origin
from util import gg_rooted_tree, gg_union
from draw import draw_common

if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = Graph(vertices=stp.nro_nodes, edges=stp.graph)

    ## DETERMINAR DUAS SOLUÇÕES PARCIAIS PELAS HEURISTICAS

    # escolher aleatoriamente um vértice terminal
    s1 = random.choice(stp.terminals)
    subtree_1, cost1 = shortest_path_with_origin(graph, s1, stp.terminals) # 0 ate 16

    s2 = random.choice(stp.terminals)
    subtree_2, cost2 = shortest_path_with_origin(graph, s2, stp.terminals)

    GU = gg_union(subtree_1, subtree_2)

    draw_common(GU, stp.terminals, subtree_1, subtree_2)

    rooted_1 = gg_rooted_tree(subtree_1, s1)
    rooted_2 = gg_rooted_tree(subtree_2, s1)


    ## PROCEDIMENTOS DE CRUZAMENTO


    all_terminals = set(stp.terminals)
    is_terminal = lambda x : x in all_terminals

    terminals = set(stp.terminals)

    std_edges = lambda x, y: (min(x,y), max(x,y))
    edges_1 = dict()
    edges_2 = dict()


    def trace_path(rooted : dict, other_graph : Graph, partition : dict, start, counter):
        v = start
        terminals_found = set()

        while rooted[v] :
            u = rooted[v]
            if is_terminal(u):
                terminals_found.add(u)

            if not other_graph.has_edge(v, u):
                partition[std_edges(v, u)] = counter

            v = rooted[v]

        return terminals_found

    counter = 0
    while terminals:
        t = terminals.pop()

        w = trace_path(rooted_1, subtree_2, edges_1, t, counter)
        w = trace_path(rooted_2, subtree_1, edges_2, t, counter)

        counter += 1


    partition_1 = dict()
    partition_2 = dict()

    for edge, p in edges_1.items():
        if not p in partition_1 :
            partition_1[p] = { 'edges' : set(), 'cost' : 0}

        weight = graph.weight(*edge)
        partition_1[p]['edges'].add(edge)
        partition_1[p]['cost'] += weight


    for edge, p in edges_2.items():
        if not p in partition_2 :
            partition_2[p] = { 'edges' : set(), 'cost' : 0}

        weight = graph.weight(*edge)
        partition_2[p]['edges'].add(edge)
        partition_2[p]['cost'] += weight
