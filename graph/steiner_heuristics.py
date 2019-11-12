# -*- coding: utf-8 -*-
import random

from graph.util.mpath import shortest_path_dijkstra as spath
from graph.graph import GraphDictionary as Graph

def an_early_shortest_path_heuristics(graph, start, terminals):
    ''' Shortest Path Heuristics (T and M)

    O que essa primeira versão está fazendo é calcular
    de uma outra forma a árvore minima MST e
    considerar somente os caminhos até os vértices terminais
    '''

    nodes = set(terminals)
    nodes.add(start)

    dist, prev = spath(graph,start)
    custo = 0

    gg = Graph()

    for u in nodes:
        while dist[u] :
            v = prev[u]
            w = graph[u][v]
            custo += w
            gg.add_edge(u,v,weight=w)
            print((u,v),' - ',w)
            u = v

    return gg, custo