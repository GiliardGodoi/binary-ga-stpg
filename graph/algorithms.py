# -*- coding: utf-8 -*-
from collections import deque
from .priorityqueue import PriorityQueue

def shortest_path_dijkstra(graph, source):
    ''' Dijkstra Algorithm - 

    Return: <vertice - distance from source> - distance dictionary from the source to vertice
            < vertice - previou > - a dictionary representing a previous node

    An early implementation has O(n^2) complexity. 
    Many improvements have been proposed across the years. 
    They are more complicated, though.

    Based in Chirantan Sharma's code
    <https://github.com/cs-oak/Fibonacci-Heaps/blob/master/dijkstra.py>
    Its use an other Priority Queue implemantation which it is not used by Sharma.
    '''
    dist = {source : 0}
    prev = {}
    done = {}

    pqueue = PriorityQueue()
    pqueue.push(0, (0, source))

    while len(pqueue):
        dist_u, u = pqueue.pop()

        if u in done :
            continue

        done[u] = True

        for v in graph.edges[u]:
            new_dist_to_v = dist_u + graph.edges[u][v]
            # esse if só da False quando: o vertice ja estiver em dist (visitado) e a distancia ja for a menor
            if (not v in dist) or (dist[v] > new_dist_to_v):
                dist[v] = new_dist_to_v
                prev[v] = u
                pqueue.push(new_dist_to_v,(new_dist_to_v,v))

    return dist, prev


def prim(graph, start):
    '''
        Prim's algorithm to define the Minimum Spanning Tree.

        TO DO: 
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