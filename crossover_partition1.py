# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:58:18 2020

@author: Giliard Almeida de Godoi
"""
from os import path
import random
from collections import deque

from graph import Graph, Reader
from graph.steiner_heuristics import shortest_path_with_origin
from util import gg_rooted_tree, gg_union, check_cycle_dfs, gg_total_weight
from draw import draw_common

if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = Graph(vertices=stp.nro_nodes, edges=stp.graph)

    ## DETERMINAR DUAS SOLUÇÕES PARCIAIS PELAS HEURISTICAS

    # escolher aleatoriamente um vértice terminal
    s1 = random.choice(stp.terminals)
    subtree_1, cost1 = shortest_path_with_origin(graph, s1, stp.terminals) # 0 ate 16

    s2 = random.choice(stp.terminals)
    subtree_2, cost2 = shortest_path_with_origin(graph, s2, stp.terminals)

    GU = gg_union(subtree_1, subtree_2)

    chart = draw_common(GU,stp.terminals,subtree_1, subtree_2)

    # custos das duas soluções
    print(cost1, cost2)

    # Primeiro uma das soluções será transformada na sua forma enraizada
    rooted_tree_1 = gg_rooted_tree(subtree_1, s1)


    # uma função auxiliar para padronizar as arestas. Será necessário?
    std_edges = lambda x, y: (min(x,y), max(x,y))


    # DETERMINAR AS PARTIÇÕES DE UM DOS PAIS E OS VÉRTICES COMUNICANTES


    # Definir o conjunto de arestas que somente pertence a subtree_2
    disjoint_edges_2 = set()

    # componentes desconexas de subtree_2
    # esse grafo sera usado por DFS_with_stop para identificar de maneira mais
    # direta, os extremos (vértices comuns) de uma partição de arestas não comuns
    # do pai 2
    GGsub2 = Graph()

    # determinando somente as arestas não comum do pai 2
    for v, u in subtree_2.gen_undirect_edges():
        if not subtree_1.has_edge(v, u):
            GGsub2.add_edge(v, u)
            disjoint_edges_2.add(std_edges(v, u))


    def DFS_with_stop(start,graph : Graph):

        stack = deque()
        stack.append(start)

        visited = set([start])
        # previous = { start : None }
        previous = set()
        cost = 0

        common_vertice = list()

        while stack :
            v = stack.pop()
            visited.add(v)

            for u in GGsub2.adjacent_to(v):
                if (not u in visited) and (not u in rooted_tree_1):
                    stack.append(u)
                elif u in rooted_tree_1:
                    common_vertice.append(u)

                previous.add(std_edges(v, u))
                cost += graph.weight(v,u)

                disjoint_edges_2.discard(std_edges(v, u))

        partition = {
            'edges' : previous,
            'common' : common_vertice,
            'cost' : cost
            }
        return partition

    subsets = list()

    while disjoint_edges_2 :
        v, u = disjoint_edges_2.pop()

        # poderia ter sido
        ## if subtree_1.has_node(v) and subtree_1.has_node(u):
        if (v in rooted_tree_1) and (u in rooted_tree_1):
            tmp = {
                'edges' : {(v , u)},
                'common' : [v, u],
                'cost' : graph.weight(v, u)
            }
            subsets.append(tmp)

        elif (v in rooted_tree_1) and (not u in rooted_tree_1):
            # print('u ',u)
            tmp = DFS_with_stop(u, graph)
            subsets.append(tmp)

        elif not (v in rooted_tree_1) and (u in rooted_tree_1):
            # print('v ', v)
            tmp = DFS_with_stop(v, graph)
            subsets.append(tmp)

        else :
            tmp = DFS_with_stop(v, graph)
            subsets.append(tmp)



    # DETERMINAR QUAIS AS ARESTAS NÃO COMUN DO OUTRO PAI PODE SER COMPARADAS


    def find_uncommon_edges_from_path(rtree, a, b, subtree, graph):
        edges = set()
        std_edges = lambda x, y: (min(x,y), max(x,y))
        cost = 0

        v = a
        while rtree[v] :
            previous = rtree[v]

            if not subtree.has_edge(v, previous):
                edges.add(std_edges(v, previous))
                w = graph.weight(v, previous)
                # print(f"Verificar aresta ({v}, {previous}) - Peso  {w}")
                cost += w

            v = rtree[v]

        u = b
        while rtree[u]:
            previous = rtree[u]

            if not subtree.has_edge(u, previous):
                edges.add(std_edges(u, previous))
                w = graph.weight(u, previous)
                # print(f"Verificar aresta ({u}, {previous}) - Peso  {w}")
                cost += w

            u = rtree[u]


        return edges, cost

    partitions = dict()
    counter = 0

    for ss in subsets :
        assert len(ss['common']) >= 2, 'Vertices em comun deve ser maior ou igual a 2'

        v = ss['common'][0]
        subsets_2 = {'edges' : set(),'common' : [v], 'cost' : 0}
        for u in ss['common'][1:] :
            edges, cost = find_uncommon_edges_from_path(rooted_tree_1, v, u, subtree_2, graph)
            subsets_2['edges'].update(edges)
            subsets_2['common'].append(u)
            subsets_2['cost'] += cost

        partitions[counter] = {'sub1' : ss, 'sub2' : subsets_2 }
        counter += 1

    # REALIZAR A COMPARAÇÃO E MONTAR A SOLUÇÃO FINAL
    GG_child = Graph()

    for v, u in subtree_1.gen_undirect_edges():
        if subtree_2.has_edge(v, u):
            ww = graph.weight(v, u)
            GG_child.add_edge(v, u, weight= ww)

    for _, pp in partitions.items():
        edges = {}

        if pp['sub1']['cost'] <= pp['sub2']['cost'] :
            edges = pp['sub1']['edges']
            # print('sub1 escolhido')

        elif pp['sub2']['cost'] < pp['sub1']['cost'] :
            edges = pp['sub2']['edges']
            # print('sub2 escolhido')


        for v , u in edges:
            ww = graph.weight(v, u)
            GG_child.add_edge(v, u, weight= ww)


    has_cycle, visited_vertices = check_cycle_dfs(GG_child, s1)
    final_cost = gg_total_weight(GG_child)


    print('Has cycle ', has_cycle)
    print('Final cost ', final_cost)
