# -*- coding: utf-8 -*-
from collections import defaultdict

class Graph(object):

    def size(self):
        raise NotImplementedError("")

    def add_edge(self,v,w):
        raise NotImplementedError("")

    def add_vertice(self,w):
        raise NotImplementedError("")

    def adjacent_to(self, v):
        raise NotImplementedError("")

    def has_vertex(self,v):
        raise NotImplementedError("")

    def has_edge(self, v, w):
        raise NotImplementedError("")

    def degree(self, v, w):
        raise NotImplementedError("")

class GraphSet(Graph):
    '''
        Classe para representar um grafo.

        Baseado nos trabalhos de:

        Robert Sedgewick; Kevin Wayne; Robert Dondero
        **Introduction to Programming in Python**
        <https://introcs.cs.princeton.edu/python/home/>
        <https://introcs.cs.princeton.edu/python/45graph/graph.py.html>
    '''
    def __init__(self, vertices=None, edges=None):

        self.__edges = edges if edges else defaultdict(set)
        self.__vertices = vertices if vertices else list()

    def size(self):
        '''
            Gostaria de informar o número de vértices e de arestas
        '''
        N = len(self.__vertices)
        E = len(self.__edges)

        return tuple(N, E)

    def add_edge(self,v,w):
        if not self.has_vertex(v):
            self.add_vertice(v)
        if not self.has_vertex(w):
            self.add_vertice(w)

        if not self.has_edge(v,w):
            self.__edges[v].add(w)
            self.__edges[w].add(v)

    def add_vertice(self,v):
        if not self.has_vertex(v):
            self.__vertices.append(v)

    def adjacent_to(self,v):
        adjacents = self.__edges[v]
        yield(adjacents)

    def has_vertex(self, v):
        return v in self.__vertices

    def has_edge(self, v, w):
        if self.has_vertex(v) :
            return (w in self.__edges[v])
        return False

    def degree(self, v):
        adj = self.__edges[v]
        return len(adj)


class GraphDictionary(Graph):

    def __init__(self, vertices=None, terminals=None, edges=None):

        if isinstance(vertices,int) : 
            self.__vertices = range(1,vertices+1) ## :(
        elif isinstance(vertices,list) :
            self.__vertices = sorted(vertices)

        self.__edges = edges if edges else defaultdict(dict)
        self.__terminals = terminals if terminals else list()

    def __getitem__(self, key):
        return self.__edges[key]

    @property
    def edges(self):
        for e in self.__edges:
            yield e

    @property
    def vertices(self):
        for v in self.__vertices:
            yield v

    def size(self):
        N = len(self.__vertices)

        return N

    def add_edge(self,v,w, weight):
        if not self.has_vertex(v):
            self.add_vertice(v)
        if not self.has_vertex(w):
            self.add_vertice(w)

        if not self.has_edge(v,w):
            self.__edges[v][w] = weight
            self.__edges[w][v] = weight

    def add_vertice(self,v, is_terminal=False):
        if not self.has_vertex(v):
            if is_terminal and (not v in self.__terminals):
                self.__terminals.append(v)
            self.__vertices.append(v)

    def adjacent_to(self,v):
        adjacents = self.__edges[v]
        return iter(adjacents.keys())

    def has_vertex(self, v):
        return (v in self.__vertices)

    def has_edge(self, v, w):
        if self.has_vertex(v) :
            return (w in self.__edges[v])
        return False

    def degree(self, v):
        adj = self.__edges[v]
        return len(adj.keys())

    def weight(self, v, w):
        if self.has_edge(v,w): 
            return self.__edges[v][w]
        else:
            return 0


class GraphSparseMatrix(Graph):
    pass

class GraphCondensedMatrix(Graph):
    pass
