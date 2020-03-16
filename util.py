from collections import deque

from graph.graph import Graph


def check_cycle_dfs(graph,start):
    '''
        Verifica se existe um ciclo em um grafo a partir de um vértice.
    '''
    stack = deque()

    visited = set([start])
    prev = dict()

    stack.append(start)

    has_cycle = False

    while stack:
        v = stack.pop()
        visited.add(v)
        for u in graph.adjacent_to(v):
            if u not in visited :
                stack.append(u)
                prev[u] = v
            elif not prev[v] == u :
                has_cycle = True

    # return has_circle, visited
    return has_cycle, visited


def gg_total_weight(graph : Graph) -> int:
    ''' Retorna a soma total dos pesos das arestas do grafo'''
    total = 0
    for v,u in graph.gen_undirect_edges():
        w = graph.weight(v,u)
        total += w

    return total


def gg_edges_number(graph : Graph) -> int:
    ''' Retorna o número de arestas em um grafo'''
    nro = 0
    for _ in graph.gen_undirect_edges():
        nro += 1

    return nro


def gg_common_edges(self, other, start_node):
    '''
        Retorna as arestas em comum a dois grafos
    '''
    common_edges = set()
    queue = deque()
    nodes_done = set()

    _stantard_edge = lambda x,y : (min(x,y), max(x,y))

    for u in self.adjacent_to(start_node):
        queue.append((start_node,u))

    while queue:
        v, u = queue.pop()
        if other.has_edge(v,u):
            common_edges.add(_stantard_edge(v,u))

        nodes_done.add(v)

        for w in self.adjacent_to(u):
            if not w in nodes_done :
                queue.append((u,w))

    return common_edges


def gg_union(A : Graph, B : Graph) -> Graph:
    ''' Retorna o Grafo união de outros dois grafos '''

    C = Graph()

    for v, u in A.gen_undirect_edges():
        w = A[v][u]
        C.add_edge(v,u,weight=w)

    for v, u in B.gen_undirect_edges():
        if not C.has_edge(v,u):
            w = B[v][u]
            C.add_edge(v,u,weight=w)

    return C


def gg_rooted_tree(tree : Graph, root) -> dict:
    '''
    Represents a tree like a dictionary where the key is a vertice and the
    value is its previous parent.
    The root vertice hasn't previous parent. So its value is None.
    '''

    if not root in tree.vertices:
        raise AttributeError("<value> for root isn't a vertice for the graph")

    rrtree = dict()
    rrtree[root] = None
    queue = deque()

    for v in tree.adjacent_to(root):
        rrtree[v] = root
        queue.append(v)

    while queue:
        u = queue.popleft()
        for v in tree.adjacent_to(u):
            if not v in rrtree:
                rrtree[v] = u
                queue.append(v)

    return rrtree


def find_tree_path(rtree : dict, a, b):
    '''
    Parameters
        rtree : dict
            dicionário que representa uma árvore. Ver gg_rooted_tree method
        a, b : graph's vertices
            vértices inicial e final

    TO DO:
    - unir as duas listas
    '''

    a_to_root = [a]
    v = a
    while rtree[v]:
        a_to_root.append(rtree[v])
        v = rtree[v]

    b_to_root = [b]
    v = b
    while rtree[v]:
        b_to_root.append(rtree[v])
        v = rtree[v]

    return a_to_root, b_to_root


def gg_tree_center(tree : Graph):
    ''' Retorna os vértices que são centros de uma árvore'''

    vertices = set(tree.vertices)
    done = set()
    leaves = deque()

    for v in vertices:
        if tree.degree(v) == 1:
            leaves.append(v)

    while len(vertices) > 2:
        v = leaves.popleft()
        done.add(v)
        for u in tree.adjacent_to(v):
            if (not u in done) or (not u in leaves):
                leaves.append(u)

        vertices.discard(v)

    return vertices


def list_degree(graph : Graph):
    '''It computes the degrees from all vertices of the graph

    Parameters
        graph : GraphDictionary

    Returns
        dict <k, v>
            where k is the vertice and v is the vertice's degree
     '''
    degree = { k : len(graph[k]) for k in graph.edges.keys() }
    return degree


def max_node_degree(graph : Graph):
    '''Retorna o vértice com o maior grau de adjacência'''

    aa = list_degree(graph)
    return max(aa,key=lambda x: aa[x])


def dfs_tree(graph : Graph, start_node):
    '''
    It procedes a Deep First Search in the graph. 
    Compute a Deep First Tree wich vertice is reached for the first time.

    Parameters
        graph : GraphDictionary
        start_node : a graph's vertice

    Returns:
        main_tree : set
            Its the Deep First Tree
        secondary_branch : set
            all return edges
    '''

    main_tree = set()
    secondary_branch = set()

    def visitar(v,u, main_branch : bool):

        min_max_edge = lambda x, y : (min(x,y), max(x,y))

        if main_branch:
            main_tree.add(min_max_edge(v,u))
        else:
            secondary_branch.add(min_max_edge(v,u))

    vertices_done = set()
    stack = deque()

    def P(v):
        vertices_done.add(v) # vertice marcado
        stack.append(v) # vertice em stack

        for w in graph.adjacent_to(v):
            if not w in vertices_done:
                visitar(v, w, True)
                P(w)
            elif (w in stack) and w != stack[-2]:
                visitar(v,w, False)

        stack.pop()

    P(start_node)

    return main_tree, secondary_branch