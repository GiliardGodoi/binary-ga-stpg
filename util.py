from collections import deque

from graph.graph import GraphDictionary


def check_circles_dfs(graph,start):
    '''
        Verifica se existe um ciclo em um grafo a partir de um vértice.
    '''
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


def gg_total_weight(graph : GraphDictionary) -> int:
    ''' Retorna a soma total dos pesos das arestas do grafo'''
    total = 0
    for v,u in graph.gen_undirect_edges():
        w = graph.weight(v,u)
        total += w

    return total


def gg_edges_number(graph : GraphDictionary) -> int:
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


def gg_union(A : GraphDictionary, B : GraphDictionary) -> GraphDictionary:
    ''' Retorna o Grafo união de outros dois grafos '''

    C = GraphDictionary()

    for v, u in A.gen_undirect_edges():
        w = A[v][u]
        C.add_edge(v,u,weight=w)

    for v, u in B.gen_undirect_edges():
        if not C.has_edge(v,u):
            w = B[v][u]
            C.add_edge(v,u,weight=w)

    return C


def gg_rooted_tree(tree : GraphDictionary, root) -> dict:
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
    @params
    rtree : dicionário que representa uma árvore - ver gg_rooted_tree
    a - b : vértices inicial e final

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


def gg_tree_center(tree : GraphDictionary):
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


def list_degree(graph : GraphDictionary):
    ''' 
        Retorna um dicionário onde
        a chave é o vértice e o valor é o grau de adjacência do vértice
     '''
    degree = { k : len(graph[k]) for k in graph.edges.keys() }
    return degree


def max_node_degree(graph : GraphDictionary):
    '''Retorna o vértice com o maior grau de adjacência'''

    aa = list_degree(graph)
    return max(aa,key=lambda x: aa[x])
