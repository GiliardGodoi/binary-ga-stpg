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
            self.__nodes = list(range(1,vertices+1))
        elif isinstance(vertices,(set,tuple)) :
            self.__nodes = list(vertices)
        elif isinstance(vertices,list) :
            self.__nodes = vertices
        else:
            self.__nodes = list()

        self.__edges = edges if edges else defaultdict(dict)

    def __getitem__(self,key):
        return self.__edges[key]

    @property
    def edges(self):
        ''' Retorna a estrutura de dados utilizada para representar as arestas, neste caso: defaultdict(dict)'''
        return self.__edges

    @property
    def vertices(self):
        ''' Retorna um iterator para iterar sobre o cojunto de vértices '''
        return self.__nodes

    def get(self,key,std = None):
        return self.__edges.get(key,std)

    def size(self):
        ''' Retorna o número de vértices no grafo '''
        return len(self.__nodes)

    def add_edge(self,v,u, weight = 1):
        '''Insere um arestas no grafo.

        @params <vértice v, vértice w, peso entre (v,w)>

        Se os vértices não existem previamente no grafo, então eles são inseridos.
        Não permite a inserção de loços, isto é, quando v == w.
        Se o parâmetro peso não é definido, é tomado como sendo de valor 1 (um)
        '''
        if v == u :
            return
        if not self.has_node(v):
            self.add_node(v)
        if not self.has_node(u):
            self.add_node(u)

        if not self.has_edge(v,u):
            self.__edges[v][u] = weight
            self.__edges[u][v] = weight

    def add_node(self,v):
        ''' @param <vértice>
        Insere um novo vértice ao conjunto de vértices. Não permite a inserção de um vértice pré-existente
        '''
        if not self.has_node(v):
            self.__nodes.append(v)

    def adjacent_to(self,v):
        ''' @param <vértice>
        Retorna um objeto <iterator> com os vértices adjacentes ao vértice passado como parâmetro.
        Se não existe arestas com o vértice informado, um <KeyError> é lançado. Não faz essa verificação.
        '''
        return self.__edges[v].keys()

    def has_node(self, v):
        ''' Verifica se um vértice existe no grafo'''
        return (v in self.__nodes)

    def has_edge(self, v, w):
        ''' Verifica se uma aresta existe no grafo '''
        if self.has_node(v) :
            return (w in self.__edges[v])
        return False

    def degree(self, v):
        ''' Retorna o grau de conexões de um vértice '''
        adj = self.__edges[v]
        return len(adj.keys())

    def weight(self, v, w):
        ''' Retorna o peso de uma aresta. Se a aresta não existe é retornado o valor 0 '''
        if self.has_edge(v,w): 
            return self.__edges[v][w]
        elif (v == w) and (v in self.__nodes) :
            return 0
        else:
            return float("inf")