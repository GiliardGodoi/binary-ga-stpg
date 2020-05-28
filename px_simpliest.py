from collections import deque

from graph import Graph
from genetic.base import Operator
from genetic.chromosome import TreeBasedChromosome

class SimplePartitionCrossover (Operator):
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

    def operation(self, PARENT_A : TreeBasedChromosome, PARENT_B : TreeBasedChromosome):
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
        # AandB_vertices = A_vertices.union(B_vertices)
        partitions = list()

        while AandB_vertices:
            start = AandB_vertices.pop()
            partition, visited = self.DFS(graph_partition,SUBTREE_A, SUBTREE_B, start)

            if partition["A"]["cost"] <= partition["B"]["cost"] :
                partitions.append(partition["A"])
            else :
                partitions.append(partition["B"])

            AandB_vertices = AandB_vertices.difference(visited) # estao em AandB_vertices mas não estão em visited O(n + m)

        for partition in partitions :
            for v, u in partition["edges"]:
                graph_common.add_edge(v, u, weight=self.graph.weight(v, u))

        return TreeBasedChromosome(graph_common)

    def DFS(self, uncommon_graph : Graph, Atree : Graph, Btree : Graph, start):

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

        for key in partition.keys():
            if len(partition[key]["edges"]) == 0 and partition[key]["cost"] == 0:
                partition[key]["cost"] = float("inf")

        return partition, vertices_done
