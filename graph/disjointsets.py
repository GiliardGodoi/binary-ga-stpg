# -*- coding: utf-8 -*-
"""
Created on

@author: Giliard Almeida de Godoi
"""
from collections import defaultdict
from graph import Graph

class Subset():

    def __init__(self, vertice, rank=0):
        self.parent = vertice
        self.rank = rank

class DisjointSet():

    def __init__(self):
        self.subsets = defaultdict()

    def make_set(self, item):
        self.subsets[item] = Subset(item)

    def find(self, item):
        if not item in self.subsets:
            raise AttributeError()

        if self.subsets[item].parent != item :
            self.subsets[item].parent = self.find(self.subsets[item].parent)

        return self.subsets[item].parent

    def union(self, v, u):
        if self.subsets[u].rank > self.subsets[v].rank:
            self.subsets[v].parent = self.subsets[u].parent

        elif self.subsets[v].rank > self.subsets[u].rank:
            self.subsets[u].parent = self.subsets[v].parent

        else :
            self.subsets[v].parent = u
            self.subsets[u].rank += 1

def has_cycle(graph : Graph):

    ss = DisjointSet()

    for v in graph.vertices:
        ss.make_set(v)

    for v, u in graph.gen_undirect_edges():
        v_rep = ss.find(v)
        u_rep = ss.find(u)

        if v_rep == u_rep:
            return True

        ss.union(v_rep, u_rep)

    return False

if __name__ == "__main__":

    tree = Graph()

    tree.add_edge('a', 'b')
    tree.add_edge('a', 'c')
    tree.add_edge('a', 'd')

    tree.add_edge('b', 'e')
    tree.add_edge('b', 'f')

    tree.add_edge('c', 'g')

    tree.add_edge('d', 'h')
    tree.add_edge('d', 'i')
    tree.add_edge('d', 'j')

    tree.add_edge('j', 'k')
    tree.add_edge('j', 'l')

    # tree.add_edge('g', 'a') ## has a cycle

    cycle = has_cycle(tree)

    print(cycle)

