# -*- coding: utf-8 -*-
from collections import deque

def bfs(graph, start=None):
    '''
    ::Breadth First Search ::

    Retorno: um conjunto do tipo <set> com os vértices encontrados pela busca em largura,
    a partir do vértice de inicío representado pelo parâmetro start.

    Se <start> não está definido no conjunto de arestas do grafo, então é lançado um
    KeyError.
    '''
    if not start:
        raise AttributeError("Start is not defined")
    elif not (start in graph.vertices):
        raise KeyError("start node is not in graph")     

    
    visited_nodes = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        visited_nodes.add(node)
        for v in graph.adjacent_to(node):
            if not v in visited_nodes:
                queue.append(v)

    return visited_nodes

def dfs(graph, start = None):
    ''' Deep First Search'''

    if not start :
        raise AttributeError('Start is not defined')
    elif not (start in graph):
        raise KeyError("Start node is not in graph")

    vertices_done = set()
    stack = deque([start])

    while stack:
        node = stack.pop()
        vertices_done.add(node)
        for v in graph.adjacent_to(node):
            if not v in vertices_done:
                stack.append(v)

    return vertices_done

def find_connected_components(graph):
    '''
        Determina as componentes conexas de um grafo dado.
        Implementação baseada em recursão e busca em largura.

        Retorno: uma lista <list> com os componentes encontrados. 
        Cada componente é representado por um conjunto <set> de vértices que pertencem àquele conjunto.

        Por definição os conjuntos encontrados devem possuir interseção vazia.
    '''
    all_nodes = set(graph.edges.keys())

    if not all_nodes:
        return {}

    def find_by_recursion(graph, start = None, nodes = None):
        if not start:
            return []
        if not nodes :
            return []

        visited = bfs(graph,start=start)

        not_visited = nodes - visited

        components = [visited]
        
        if len(not_visited):
            n_start = not_visited.pop()
            components += find_by_recursion(graph,start=n_start,nodes=not_visited)

        return components

    start = all_nodes.pop()

    return find_by_recursion(graph,start=start,nodes=all_nodes)

