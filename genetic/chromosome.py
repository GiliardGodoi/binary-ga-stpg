# -*- coding: utf-8 -*-

class Chromosome(object):

    def __init__(self,tree, fitness):
        self.tree_graph = tree
        self.fitness = fitness
        self.score = fitness 