import random

from genetic.base import BaseGA
from genetic.chromosome import BinaryChromosome, TreeBasedChromosome
from graph import Graph, SteinerTreeProblem
from graph.disjointsets import DisjointSets
from graph.priorityqueue import PriorityQueue
from graph.util import has_cycle, how_many_components

#######################################################################
# GENERATE A RANDOM CHROMOSOME
#######################################################################

def random_treegraph_chromosome(graph : Graph, terminals):
    '''
    Parameters:
        graph : Graph
            Grafo que serve de instância para o problema
    '''
    alledges = [(v, u, graph.weight(v,u)) for v, u in graph.gen_undirect_edges()]
    random.shuffle(alledges) # random component

    terminals = set(terminals)
    DS = DisjointSets()
    subgraph = Graph()
    total_cost = 0

    while alledges and terminals:
        v, u, weight = alledges.pop()

        if v not in DS:
            DS.make_set(v)
        if u not in DS:
            DS.make_set(u)

        if DS.find(v) != DS.find(u):
            DS.union(v, u)
            total_cost += weight
            subgraph.add_edge(v, u, weight=weight)
            terminals.discard(v)
            terminals.discard(u)

    return TreeBasedChromosome(subgraph)

#######################################################################
# EVALUATING
#######################################################################

def vertices_from_binary_chromosome(chromosome, terminals, nro_vertices):

    terminals = set(terminals)
    vertices = set()
    genes = chromosome.genes
    index = len(genes) - 1

    for v in range(nro_vertices, 0, -1):
        if v in terminals:
            vertices.add(v)
        else:
            if genes[index] == '1':
                vertices.add(v)
            index -= 1

    return vertices

def evaluate_treegraph(chromosome, penality):

    if type(chromosome) is not TreeBasedChromosome:
        raise TypeError("chromosome is not what was expected")

    total_cost = 0
    qtd_partition = 0
    DS = DisjointSets()

    for v in chromosome.graph.vertices:
        DS.make_set(v)

    for v, u in chromosome.graph.gen_undirect_edges():
        if DS.find(v) == DS.find(u):
            print("FOI IDENTIFICADO UM CICLO EM UMA DAS SOLUÇÕES")
        DS.union(v,u)
        total_cost +=  chromosome.graph.weight(v, u)

    qtd_partition = len(DS.get_disjoint_sets())

    total_cost += penality(qtd_partition)

    return total_cost, qtd_partition > 1

def evaluate_binary(chromosome, GRAPH, terminals, nro_vertices, penality):

        # instânciando variáveis e funções auxiliares
        queue = PriorityQueue()
        DS = DisjointSets()
        dones = set()

        ## identifica todos os vértices não terminais representados no cromossomo
        vertices = vertices_from_binary_chromosome(chromosome, terminals, nro_vertices)

        # adiciona uma aresta se os vértices extremos da aresta
        # estão contidos no conjunto vertices
        # mantém essas arestas em uma fila de prioridades para
        # formar uma MST baseado no algoritmo de Kruskal
        # (o trabalho de Kapsalis utilizava o algoritmo de Prim)
        for v in vertices:
            dones.add(v)
            DS.make_set(v) # para construir a MST
            for u in GRAPH.adjacent_to(v):
                if (u in vertices) and (u not in dones):
                    weight = GRAPH.weight(v,u)
                    queue.push(weight, (v, u, weight))

        total_cost = 0
        while queue:
            v, u, weight = queue.pop()
            if DS.find(v) != DS.find(u):
                total_cost  += weight
                DS.union(v, u)
            # Repare que não construimos a MST mas apenas
            # definimos os conjuntos disjuntos.

        qtd_partition = len(DS.get_disjoint_sets())

        total_cost += penality(qtd_partition)

        return total_cost, qtd_partition > 1

#######################################################################
# CONVERTING
#######################################################################

def convert_treegraph2binary(chromosome, terminals, nro_vertices):
    '''Converts from TreeChromosome to a BinaryChromosome'''

    if type(chromosome) is not TreeBasedChromosome:
        raise TypeError("chromosome is not what was expected")

    terminals = set(terminals)

    subgraph = chromosome.genes
    genes = ['0'] * nro_vertices # all vertices in the instance problem

    # vertices in the subgraph (partial solution) include terminals and non-terminals
    for v in subgraph.vertices:
        genes[v-1] = '1'

    # choosing only the non_terminals positions
    genes = (gene for v, gene in enumerate(genes, start=1) if v not in terminals)

    return BinaryChromosome(''.join(genes))

def convert_binary2treegraph(chromosome, GRAPH, terminals, nro_vertices):
    '''Converts from BinaryChromosome to TreeChromosome'''

    if type(chromosome) is not BinaryChromosome:
        raise TypeError("chromosome is not what was expected")

    queue = PriorityQueue()
    disjointset = DisjointSets()
    subgraph = Graph()
    dones = set()

    # todos os vértices esperados na solução parcial
    vertices = vertices = vertices_from_binary_chromosome(chromosome, terminals, nro_vertices)

    # adiciona uma aresta se os vértices extremos da aresta
    # estão contidos no conjunto vertices
    # mantém essas arestas em uma fila de prioridades para
    # formar uma MST baseado no algoritmo de Kruskal
    for v in vertices:
        dones.add(v)
        disjointset.make_set(v) # para construir a MST
        subgraph.add_node(v) # garantir a inserção de um vértice isolado
        for u in GRAPH.adjacent_to(v):
            if (u in vertices) and (u not in dones):
                weight = GRAPH.weight(v, u)
                queue.push(weight, (v, u, weight))

    while queue:
        v, u, weight = queue.pop()
        if disjointset.find(v) != disjointset.find(u):
            subgraph.add_edge(v, u, weight=weight)
            disjointset.union(v, u)

    return TreeBasedChromosome(subgraph)

#######################################################################
# PRINTING AND SAVING
#######################################################################

