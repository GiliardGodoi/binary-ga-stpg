# -*- coding: utf-8 -*- 
from collections import defaultdict

class GraphDictionary(object):
    '''
        Classe para representar um grafo.

        Baseado nos trabalhos de:

        Robert Sedgewick; Kevin Wayne; Robert Dondero
        **Introduction to Programming in Python**
        <https://introcs.cs.princeton.edu/python/home/>
        <https://introcs.cs.princeton.edu/python/45graph/graph.py.html>
    '''
    def __init__(self, vertices=None, edges=None):

        if isinstance(vertices,int) : 
            self.__nodes = range(1,vertices+1) ## :(
        elif isinstance(vertices,(list,set,tuple)) :
            self.__nodes = sorted(vertices)
        else:
            self.__nodes = list()

        self.__edges = edges if edges else defaultdict(dict)

    def __getitem__(self,key):
        return self.__edges[key]

    @property
    def edges(self):
        return self.__edges

    @property
    def vertices(self):
        return iter(self.__nodes)

    def size(self):
        return len(self.__nodes)

    def add_edge(self,v,w, weight):
        if not self.has_node(v):
            self.add_node(v)
        if not self.has_node(w):
            self.add_node(w)

        if not self.has_edge(v,w):
            self.__edges[v][w] = weight
            self.__edges[w][v] = weight

    def add_node(self,v):
        if not self.has_node(v):
            self.__nodes.append(v)

    def adjacent_to(self,v):
        adjacents = self.__edges[v]
        return iter(adjacents.keys())

    def has_node(self, v):
        return (v in self.__nodes)

    def has_edge(self, v, w):
        if self.has_node(v) :
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