# -*- coding: utf-8 -*-
import random

from graph.util.mpath import shortest_path_dijkstra as minPath
from graph.util.mst import prim
from graph.graph import GraphDictionary as Graph

def minimum_paths_heuristic(graph, start, terminals):
    ''' Minimum path

    Calcula a Árvore de Caminhos Mínimos de um nó origem S <start> até os demais nós.
    Realiza uma poda considerando os nós termianis como folhas até a raiz S start.

    Vale lembrar que a árvore de caminhos mínimos representa os caminhos mínimos de um vértice S
    até todos os outros vértices. 
    '''

    dist, prev = minPath(graph,start)
    custo = 0

    stree = Graph()

    for u in terminals:
        while dist[u] :
            v = prev[u]
            w = graph[u][v]
            custo += w
            stree.add_edge(u,v,weight=w)
            u = v

    return stree, custo

def shortest_path_heuristic(graph, start, terminals):
    ''' Adaptação para o algortimo Shortest Path Heuristic

    A Árvore solução Tsph é contruida iterativamente: um vertice terminal é incluido por vez.

    No algoritmo original, no passo para incluir um novo vértice é considerado
    o menor caminho entre o vértice terminal que se deseja incluir e a árvore que se está construindo.
    Em termos práticos é considerado a menor distância que liga qualquer um dos vértices que pertence à árvore Tsph
    e o novo vértice que se deseja incluir.
    Esse procedimento requer que seja calculada a matriz de caminhos mínimos de todos para todos os vértices.
    Isso pode ser feito pelos Algortimos de Floyd-Warshall ou pelo Johnson (Ver o livro de CRLS)

    O que se faz aqui é calcular Árvore de caminhos mínimos de um ponto inicial até os demais vértices,
    realizar a poda (como na função anterior) para determinar um subconjunto de vértices.
    São consideradas então somente as arestas dos vértices desse subconjunto de vértices.
    Então é calculada a MST desse subgrafo.
    '''

    dist, prev = minPath(graph,start)

    selectedNodes = set([start])

    for t in terminals:
        selectedNodes.add(t)
        u = t
        while dist[u]:
            v = prev[u]
            selectedNodes.add(v)
            u = v

    subgraph = Graph()

    for v in selectedNodes:
        for u in graph.adjacent_to(v):
            if (u in selectedNodes) :
                w = graph.edges[v][u]
                subgraph.add_edge(v,u,weight=w)

    subtree, custo = prim(subgraph,start)

    return subtree, custo

def mst_prunning_heuristic(graph, terminals, start):
    '''
        Determina a MST do grafo por meio do algoritmo de Prim.
        Realiza a poda considerando os nós terminais como os nós folhas até a raiz <start>
        Se <start> é um nó terminal o peso da árvore será sempre igual.
    '''
    mst, _ = prim(graph,start)

    mst[start] = False

    gg = Graph()

    total_weight = 0 

    for t in terminals:
        node = t
        while mst[node]:
            prev = mst[node]
            if gg.has_edge(prev,node):
                break # já foi inserido esse ramo
            weight = graph[prev][node]
            total_weight += weight
            gg.add_edge(prev,node,weight=weight)
            node = prev

    return gg, total_weight