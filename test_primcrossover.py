import pprint as pp
from os import path
from collections import deque
from graph.priorityqueue import PriorityQueue

from graph.graph import GraphDictionary
from graph.steiner_heuristics import shortest_path_origin_prim as heuristic
from graph.steiner_heuristics import prunning_mst


from graph.reader import Reader


def check_circles_dfs(graph,start):
    stack = deque()

    visited = set([start])
    prev = dict()

    stack.append(start)

    has_circle = False

    while stack:
        v = stack.pop()
        visited.add(v)
        for u in graph.adjacent_to(v):
            if u not in visited :
                stack.append(u)
                prev[u] = v
            elif not prev[v] == u :
                has_circle = True

    return has_circle,visited

def gg_union(A : GraphDictionary, B : GraphDictionary) -> GraphDictionary:
    

    C = GraphDictionary()

    for v, u in A.gen_undirect_edges():
        w = A[v][u]
        C.add_edge(v,u,weight=w)

    for v, u in B.gen_undirect_edges():
        if not C.has_edge(v,u):
            w = B[v][u]
            C.add_edge(v,u,weight=w)

    return C


if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = GraphDictionary(vertices=stp.nro_nodes, edges=stp.graph)

    sub1, c1 = heuristic(graph, stp.terminals[8], stp.terminals)
    sub2, c2 = heuristic(graph, stp.terminals[1], stp.terminals)

    GU = gg_union(sub1,sub2)

    sub3, c3 = prunning_mst(GU, stp.terminals[1], stp.terminals)