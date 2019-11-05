# -*- coding: utf-8 -*-

class Chromosome(object):

    def __init__(self,genes = None, size = None, fitness = None):

        if genes :
            self.__genes = genes
            self.__size = len(genes)
        elif size and not genes :
            self.__size = size
            self.__genes = dict()
        else :
            raise TypeError("Não inicializou o cromossomo apropriadamente")

        self.__fitness = fitness if fitness else 0
        self.__score = fitness  if fitness else 0
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

    def __getitem__(self, index):
        ''' It allows the chromosome to be accessible by index '''
        return self.__genes[index]

    def __setitem__(self,index, value):
        self.__genes[index] = value

    def __len__(self):
        ''' It returns the size of genes' array '''
        return self.__size

    def __str__(self):
        tmp = [str(self.__genes[i]) for i in range(0,3) ]
        template = ' '.join(tmp)
        return f'[{template}... ]'

    def __repr__(self):
        return self.__str__()