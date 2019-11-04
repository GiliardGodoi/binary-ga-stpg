# -*- coding: utf-8 -*-

class Chromosome(object):

    def __init__(self,tree, fitness):

        self.genes = tree

        self.__fitness = fitness
        self.__score = fitness 
        self.was_normalized = False

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value
        self.__score = value
        self.was_normalized = False

    @property
    def score(self):
        return self.__score

    @score.setter
    def score(self, value):
        self.__score = value
        self.was_normalized = True


    def __lt__(self,chromo) : 
        ''' It implements < comparison operator '''
        return self.__score < chromo.__score

    def __le__(self, chromo):
        ''' It implements <= comparison operator '''
        return self.__score <= chromo.__score

    def __eq__(self, chromo):
        ''' It implements == comparison operator '''
        return self.__score == chromo.__score

    def __ne__(self, chromo):
        ''' It implements != comparison operator '''
        return self.__score != chromo.__score

    def __ge__(self, chromo):
        '''It implements >= comparison operator '''
        return self.__score >= chromo.__score

    def __gt__(self,chromo):
        '''It implements > comparison operator '''
        return self.__score > chromo.__score