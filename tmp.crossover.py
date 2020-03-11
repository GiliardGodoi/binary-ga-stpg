import pprint as pp
from collections import deque
from os import path
#import string

from graph import GraphDictionary, Reader
from graph.steiner_heuristics import shortest_path_with_origin
from graph.algorithms import find_connected_components

from draw import draw_common

diretorio_dados = "datasets"
arquivo_dados = "b13.stp"
arquivo = path.join(diretorio_dados, arquivo_dados)

reader = Reader()

stp = reader.parser(arquivo)

graph = GraphDictionary(vertices=stp.nro_nodes,edges=stp.graph)

#duas soluções parciais
sub1, c1 = shortest_path_with_origin(graph,stp.terminals[5],stp.terminals) # 2, 0
sub2, c2 = shortest_path_with_origin(graph,stp.terminals[4],stp.terminals) # 5, 8e

######################################################################################
##############                           TESTE 1                        ##############
######################################################################################

# s = stp.terminals[0]
# # s = 18

# terminals = set(stp.terminals)

# pazul = set()
# pv_azul = set([s])

# pvermelho = set()
# pv_vermelho = set([s])
# principal_vermelho = set([s])


# common = set()

# stack = deque()

# for u in sub2.adjacent_to(s):
#     stack.append((s,u))

# del u

# while stack:
#     v, u = stack.pop()
#     if not sub1.has_edge(v,u):
#         pvermelho.add((v,u))
#         for a in sub2.adjacent_to(u):
#             if not a in pv_vermelho:
#                 stack.append((u,a))
#         pv_vermelho.add(u)
#         if u in terminals :
#             principal_vermelho.add(u)
#     else :
#         principal_vermelho.add(v)
#         common.add((v,u))

# regularizar = lambda x, y : (min(x,y), max(x,y))

# E1 = set([regularizar(*e) for e in sub1.gen_undirect_edges()])

# E2 = set([regularizar(*e) for e in sub2.gen_undirect_edges()])

# common = E1 & E2
# onlyE1 = E1 - E2
# onlyE2 = E2 - E1

# print(common,'\n\n')

# gg = GraphDictionary()

# for e in common:
#     v,u = e
#     gg.add_edge(v,u,weight=stp.graph[v][u])

# common_components = find_connected_components(gg)

# print(common_components)

# print(stp.graph[18][21])
# print(stp.graph[22][21])

# gg.add_edge(22,21,weight=stp.graph[22][21])


######################################################################################
##############                           TESTE 2                        ##############
######################################################################################


# G_child = GraphDictionary()
# G_candidates = GraphDictionary()
# G_sub1 = GraphDictionary()
# G_sub2 = GraphDictionary()

# # Criar um grafo união e remover as arestas em comuns
# for v,u in sub1.gen_undirect_edges():
#     if sub2.has_edge(v,u) :
#         G_child.add_edge(u,v,weight=graph.weight(v,u))
#     else :
#         G_sub1.add_edge(u,v,weight=graph.weight(v,u))
#         G_candidates.add_edge(u,v,weight=graph.weight(v,u))

# for v,u in sub2.gen_undirect_edges() :
#     if sub1.has_edge(v,u):
#         #já foram adicionadas a G_child
#         continue
#     else :
#         G_sub2.add_edge(u,v,weight=graph.weight(v,u))
#         G_candidates.add_edge(u,v,weight=graph.weight(v,u))

# components_conected = find_connected_components(G_child)
# candidates = find_connected_components(G_candidates)



# if len(components_conected) <= len(string.ascii_uppercase):
#     pass
# else:
#     raise "Erro mas ok"

# i = 0
# components = dict()

# for cc in components_conected :
#     k = string.ascii_uppercase[i]
#     components[k] = cc
#     i += 1

# def map_components(vertice):
#     for key, value in components.items():
#         if vertice in value:
#             return key


# for v in candidates[0] :
#     esta_pai1 = sub1.has_node(v)
#     esta_pai2 = sub2.has_node(v)

#     if esta_pai1 and not esta_pai2:
#         print(v,'  - pai 1')
#     elif esta_pai2 and not esta_pai1:
#         print(v,'  - PAI 2')
#     elif esta_pai1 and esta_pai2:
#         code = map_components(v)
#         print(v,'  - AMBOS OS PAIS  ',code)
#     else :
#         print('ERROR')

# vertices_comunicantes = set()

# for v in candidates[0] :
#     esta_pai1 = sub1.has_node(v)
#     esta_pai2 = sub2.has_node(v)

#     if esta_pai1 and esta_pai2:
#         vertices_comunicantes.add(v)

# pre_tree = dict()


######################################################################################
##############                           TESTE 3                        ##############
######################################################################################

# child = GraphDictionary()
# queue = deque()
# nonCommonSub1 = deque()
# start_node = stp.terminals[0]

# nodes_done = set()

# for u in sub1.adjacent_to(start_node):
#     queue.append((start_node,u))

# while queue:
#     v, u = queue.pop()
#     if sub2.has_edge(v,u):
#         child.add_edge(v,u)
#     else:
#         nonCommonSub1.append((v,u))

#     nodes_done.add(v)

#     for w in sub1.adjacent_to(u):
#         if not w in nodes_done :
#             queue.append((u,w))

# # a fila (queue) está vazia
# # Nessa altura eu não sei quem são os vértices comunicantes

# nonCommonSub2 = deque()
# nodes_visited = set()

# for u in sub2.adjacent_to(start_node):
#     queue.append((start_node,u))

# while queue:
#     v, u = queue.pop()
#     if sub1.has_edge(v,u):
#         pass
#     else :
#         nonCommonSub2.append((v,u))

#     nodes_visited.add(v)

#     for w in sub2.adjacent_to(u):
#         if not w in nodes_visited:
#             queue.append((u,w))


######################################################################################
##############                           TESTE 4                        ##############
######################################################################################

def common_edges(stree1,other, start_node):
    commonEdges = list()
    nonCommon = list()
    queue = deque()
    nodes_done = set()

    for u in stree1.adjacent_to(start_node):
        queue.append((start_node,u))

    while queue:
        v, u = queue.pop()
        if other.has_edge(v,u):
            commonEdges.append((v,u))
        else:
            nonCommon.append((v,u))

        nodes_done.add(v)

        for w in sub1.adjacent_to(u):
            if not w in nodes_done :
                queue.append((u,w))

    return commonEdges, nonCommon

def find_components_from(edges_list):

    u, v = edges_list.pop(0)
    components =  list()
    components.append(set([u,v]))
    i = 0
    while len(edges_list):
        u, v = edges_list.pop(0)
        if u in components[i]:
            components[i].add(v)
        elif v in components[i]:
            components[i].add(u)
        else:
            components.append(set([u,v]))
            i += 1

    return components

def tree_rooted(tree, root):
    '''
    Represents a tree like a dictionary where the key is a vertice and the 
    value is its previous parent.
    The root vertice hasn't previous parent. So its value is None.
    '''

    if not root in tree.vertices:
        raise AttributeError("<value> for root isn't a vertice for the graph")

    dicttree = dict()
    dicttree[root] = None
    queue = deque()

    for v in tree.adjacent_to(root):
        dicttree[v] = root
        queue.append(v)

    while queue:
        u = queue.popleft()
        for v in tree.adjacent_to(u):
            if not v in dicttree:
                dicttree[v] = u
                queue.append(v)

    return dicttree


def find_tree_path(rtree,a,b):

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

def list_degree(graph):
    degree = { k : len(graph[k]) for k in graph.edges.keys() }
    return degree

def max_node_degree(graph):

    aa = list_degree(graph)
    return max(aa,key=lambda x: aa[x])

# def tree_center(tree : GraphDictionary) :

#     leaves = deque()
#     for v in tree.vertices:
#         if tree.degree(v) == 1:
#             leaves.append(v)
            
#     while tree.size() > 2:
#         v = leaves.popleft()
#         for u in tree.adjacent_to(v):
#             leaves.append(u)
        
#         tree.remove_node(v)

#     return tuple(tree.edges.keys())

def tree_center(tree : GraphDictionary):

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

def graph_weight(graph : GraphDictionary):

    total = 0
    for v,u in graph.gen_undirect_edges():
        w = graph.weight(v,u)
        total += w

    return total

def number_edges(graph : GraphDictionary):

    nro = 0
    for _ in graph.gen_undirect_edges():
        nro += 1

    return nro


def st_partition_crossover(Atree : GraphDictionary, Btree : GraphDictionary):
    ''' Operação de cruzamento para o STPG '''

    child = GraphDictionary()
    GT = GraphDictionary()

    for v,u in Atree.gen_undirect_edges():
        w = Atree.weight(v,u)
        if Btree.has_edge(v,u):
            child.add_edge(v,u,weight=w)
        else:
            GT.add_edge(v,u,weight=w)

    for v,u in Btree.gen_undirect_edges():
        if not Atree.has_edge(v,u):
            GT.add_edge(v,u)

    return child, GT

def st_union_tree(Atree : GraphDictionary, Btree : GraphDictionary):
    ''' Operação de cruzamento para o STPG '''

    child = GraphDictionary()

    for v,u in Atree.gen_undirect_edges():
        w = Atree.weight(v,u)
        child.add_edge(v,u,weight=w)

    for v,u in Btree.gen_undirect_edges():
        w = Btree.weight(v,u)
        child.add_edge(v,u,weight=w)

    return child

######################################################################################
##############                           TESTE 5                        ##############
######################################################################################

def update_partition(start, end, partition : dict):

    if start is None or end is None:
        print('ERROR  ::',start,end)
        return
    edge = (min(start, end), max(start,end))

    if not start in partition:
        partition[end] = set()
        partition[end].add(edge)

    else:
        tmp = set(partition[start]) # Operação de atribuiçao implica em cópia
        tmp.add(edge)
        partition[end] = tmp
    # partition.add(edge)

# A_partition = dict()
# B_partition = dict()

A_partition = dict()
B_partition = dict()

child = GraphDictionary()
GU = GraphDictionary()

for v, u in sub1.gen_undirect_edges():
    GU.add_edge(v,u)

for v, u in sub2.gen_undirect_edges():
    if not GU.has_edge(v,u):
        GU.add_edge(v,u)

start_node = max(stp.terminals,key=lambda v : GU.degree(v))

stack = deque()

done_red = set([start_node])
done_blue = set([start_node])


for adj in GU.adjacent_to(start_node):
    stack.append((start_node, adj))


while stack :
    v, u = stack.pop()

    if sub1.has_edge(v,u) and sub2.has_edge(v,u):
        w = sub1.weight(v,u)
        print('both:  ',v,u)
        child.add_edge(v,u,weight=w)

        done_red.add(u)
        done_blue.add(u)

        for next_nd in GU.adjacent_to(u):
            if next_nd in done_blue and next_nd in done_red:
                continue
            else:
                stack.append((u, next_nd))

    elif sub1.has_edge(v,u) and not sub2.has_edge(v,u):
        print('A:  ',v,u)
        update_partition(v,u,A_partition)

        done_blue.add(u)

        for next_nd in sub1.adjacent_to(u):
            if next_nd in done_blue:
                continue
            else:
                stack.append((u, next_nd))


    elif not sub1.has_edge(v,u) and sub2.has_edge(v,u):
        print('B:  ',v,u)
        update_partition(v,u,B_partition)

        done_red.add(u)

        for next_nd in sub2.adjacent_to(u):
            if next_nd in done_red:
                continue
            else:
                stack.append((u, next_nd))

    else:
        print("ERROR",v,u)


print('\n\n\nA:\n')

pp.pprint(A_partition)

print('\n\n\nB:\n')

pp.pprint(B_partition)

tt = stp.terminals

# draw_common(GU,tt,sub2,sub1)


########################################

GUp = {}

leaves = set()
    
for k in GUp.keys():
    leaves.add(k)

for vl in GUp.values():
    leaves.discard(vl)