# -*- coding: utf-8 -*-
from collections import deque

def find_path(graph, start, end):
    path = list()
    return __find_path_iterativamente(graph,start,end,path)


def __find_path_iterativamente(graph, start, end, path = []):
    '''
        Função baseada na implementação encontrada em: 
        <https://www.python.org/doc/essays/graphs/> Acessado em 05.11.2019
    '''
    path += [start]

    if start == end :
        return path
    if not graph.has_vertex(start) :
        return None

    for node in graph.adjacent_to(start):
        if not (node in path) :
            new_path = __find_path_iterativamente(graph,node,end, path)
            if new_path:
                return new_path

    return None

def find_shortest_path(graph, start, end):
    '''
        Método baseado na implementação de Eryk Kopczyński
        <https://www.python.org/doc/essays/graphs/> Acessado em 05.11.2019
    '''
    dist = {start: [start]}
    q = deque()
    q.append(start)
    while len(q):
        at = q.popleft()
        for next in graph.adjacent_to(at):
            if next not in dist:
                dist[next] = [dist[at], next]
                q.append(next)
    return dist[end]
