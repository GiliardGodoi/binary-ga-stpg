# -*- coding: utf-8 -*-
from collections import defaultdict

class Graph(object):

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

        self.edges = edges if edges else defaultdict(set)
        self.vertices = vertices if vertices else list()

    def add_edge(self,v,w):
        if not self.has_vertex(v):
            self.add_vertice(v)
        if not self.has_vertex(w):
            self.add_vertice(w)

        if not self.has_edge(v,w):
            self.edges[v].add(w)
            self.edges[w].add(v)

    def add_vertice(self,v):
        if not self.has_vertex(v):
            self.vertices.append(v)

    def adjacent_to(self,v):
        adjacents = self.edges[v]
        return iter(adjacents)

    def has_vertex(self, v):
        return v in self.vertices

    def has_edge(self, v, w):
        if self.has_vertex(v) :
            return (w in self.edges[v])
        return False

    def degree(self, v):
        adj = self.edges[v]
        return len(adj)


class GraphDictionary(Graph):

    def __init__(self, vertices=None, terminals=None, edges=None):

        self.edges = edges if edges else defaultdict(dict)
        self.vertices = vertices if vertices else list()
        self.terminals = terminals if terminals else list()

    def add_edge(self,v,w, weight):
        if not self.has_vertex(v):
            self.add_vertice(v)
        if not self.has_vertex(w):
            self.add_vertice(w)

        if not self.has_edge(v,w):
            self.edges[v][w] = weight
            self.edges[w][v] = weight

    def add_vertice(self,v, is_terminal=False):
        if not self.has_vertex(v):
            if is_terminal and (not v in self.terminals):
                self.terminals.append(v)
            self.vertices.append(v)

    def adjacent_to(self,v):
        adjacents = self.edges[v]
        return iter(adjacents.keys())

    def has_vertex(self, v):
        return v in self.vertices

    def has_edge(self, v, w):
        if self.has_vertex(v) :
            return (w in self.edges[v])
        return False

    def degree(self, v):
        adj = self.edges[v]
        return len(adj.keys())

    def weight(self, v, w):
        if self.has_edge(v,w): 
            return self.edges[v][w]
        else:
            return 0


class GraphSparseMatrix(Graph):
    pass

class GraphCondensed(Graph):
    pass