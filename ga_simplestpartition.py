# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:26:16 2020

@author: Giliard Almeida de Godoi
"""
from os import path
import random
from collections import deque

from graph import Graph, ReaderORLibrary
from graph.steiner_heuristics import shortest_path_with_origin
from graph.util import has_cycle, gg_total_weight

class Chromosome(object):

    def __init__(self, subtree : Graph):
        self.graph = subtree
        self.__cost = 0
        self.__fitness = 0
        self.normalized = False

    @property
    def cost(self):
        return self.__cost

    @cost.setter
    def cost(self, value):
        self.__cost = value
        self.__fitness = value
        self.normalized = False

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value
        self.normalized = True

    def __len__(self):
        return len(self.graph.vertices)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def __repr__(self):
        return self.__str__()

class SimplePartitionCrossover(object):
    '''
    Operação de Cruzamento baseado em avaliação das partições.

    1. Identifica as arestas comuns e as arestas não comuns, formando componentes conexas.
    2. Arestas comuns são transmitidas para os descendentes.
    3. Arestas não comuns formam partições (componentes conexas) que serão avaliadas isoladamente.
    4. Para cada partição (passo 3) compara o peso das arestas de cada um dos pais.
    O subconjunto de arestas de menor peso (custo) será transmitido ao descendente.

    Notes:
    Este procedimento não garante que todos os descendentes serão factíveis.
    '''

    def __init__(self, graphinstance : Graph):
        self.graph = graphinstance

    def crossing(self, PARENT_A : Chromosome, PARENT_B : Chromosome):
        '''Implementa a operação de crossover'''

        SUBTREE_A = PARENT_A.graph
        SUBTREE_B = PARENT_B.graph

        graph_common = Graph()
        graph_partition = Graph()

        A_vertices = set()
        B_vertices = set()

        for v, u in SUBTREE_A.gen_undirect_edges():
            weight = SUBTREE_A.weight(v,u)
            if SUBTREE_B.has_edge(v,u):
                graph_common.add_edge(v, u, weight=weight)
            else:
                A_vertices.add(v)
                A_vertices.add(u)
                graph_partition.add_edge(v, u, weight=weight)

        for v, u in SUBTREE_B.gen_undirect_edges():
            weight = SUBTREE_B.weight(v, u)
            if not SUBTREE_A.has_edge(v, u):
                B_vertices.add(v)
                B_vertices.add(u)
                graph_partition.add_edge(v, u, weight=weight)

        AandB_vertices = A_vertices.intersection(B_vertices)
        partitions = list()

        while AandB_vertices:
            start = AandB_vertices.pop()
            partition, visited = self.__dfs__(graph_partition,SUBTREE_A, SUBTREE_B, start)

            if partition["A"]["cost"] <= partition["B"]["cost"] :
                partitions.append(partition["A"])
            else :
                partitions.append(partition["B"])

            AandB_vertices = AandB_vertices.difference(visited) # estao em AandB_vertices mas não estão em visited O(n + m)


        for partition in partitions :
            for v, u in partition["edges"]:
                graph_common.add_edge(v, u, weight=self.graph.weight(v, u))

        return graph_common


    def __dfs__(self, uncommon_graph : Graph, Atree : Graph, Btree : Graph, start):

        vertices_done = set()
        stack = deque([start])

        partition = {
                "A" : {"edges": set(), "cost" : 0 },
                "B" : {"edges": set(), "cost" : 0 }
            }

        while stack:
            node = stack.pop()
            vertices_done.add(node)

            for adj in uncommon_graph.adjacent_to(node):

                if adj not in vertices_done:
                    if Atree.has_edge(node, adj):
                        partition["A"]["edges"].add((node, adj))
                        partition["A"]["cost"] += uncommon_graph.weight(node, adj)
                    elif Btree.has_edge(node, adj):
                        partition["B"]["edges"].add((node, adj))
                        partition["B"]["cost"] += uncommon_graph.weight(node, adj)

                    stack.append(adj)

        return partition, vertices_done

def main():

    filename = path.join("datasets", "ORLibrary", "steinb13.txt")

    reader = ReaderORLibrary()

    STPG = reader.parser(filename)

    graph = Graph(vertices=STPG.nro_nodes, edges=STPG.graph)

    ## DETERMINAR DUAS SOLUÇÕES PARCIAIS PELAS HEURISTICAS
    # escolher aleatoriamente um vértice terminal
    s1, s2 = random.sample(STPG.terminals, k=2)

    SUBTREE_A, cost1 = shortest_path_with_origin(graph, s1, STPG.terminals) # 0 ate 16
    SUBTREE_B, cost2 = shortest_path_with_origin(graph, s2, STPG.terminals)

    SPX = SimplePartitionCrossover(graphinstance=graph)

    offspring = SPX.crossing(Chromosome(SUBTREE_A), \
                             Chromosome(SUBTREE_B))

    offspring_cost = gg_total_weight(offspring)

    print(
            f"Parent A: {cost1}\n",
            f"Parent B: {cost2}\n"
            f"Offspring: {offspring_cost}"
        )
    print("Has cycle", has_cycle(offspring))



if __name__ == "__main__":

    for i in range(50):
        main()
        print("\n========================\n")
