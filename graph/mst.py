# -*- coding: utf-8 -*-
from .priorityqueue import PriorityQueue


def prim(graph, start):
    '''
        Prim's algorithm for the MST problem.

        TODO: 
            - Verificar se para diferentes pontos de inicialização retorna a mesma árvore
            se sim, parece que está funcionando ok.
    '''
    if start not in graph.vertices:
        raise KeyError("start is not in graph vertices")

    mtree = {}
    total_weight = 0
    queue = PriorityQueue()

    queue.push(0,(start, start))

    while queue :
        node_start, node_end = queue.pop()
        if node_end not in mtree:
            mtree[node_end] = node_start
            total_weight += graph.weight(node_start, node_end)

            for next_node, weight in graph.edges[node_end].items():
                queue.push(weight,(node_end, next_node))

    return mtree, total_weight

def kruskal(graph, start):
    raise NotImplementedError("Ainda não implementado")
