import pprint as pp
from collections import deque
from os import path
from matplotlib import pyplot as plt
import networkx as nx

from graph import GraphDictionary, Reader
from graph.steiner_heuristics import shortest_path_with_origin
from graph.search import find_connected_components

diretorio_dados = "datasets"
# arquivo_dados = "b01.stp"
arquivo_dados = "b13.stp"
arquivo = path.join(diretorio_dados, arquivo_dados)

reader = Reader()

stp = reader.parser(arquivo)

MyGraph = GraphDictionary(vertices=stp.nro_nodes,edges=stp.graph)

sub1, c1 = shortest_path_with_origin(MyGraph,stp.terminals[1],stp.terminals) # 2, 0
sub2, c2 = shortest_path_with_origin(MyGraph,stp.terminals[9],stp.terminals) # 5, 8


def draw_graph(graph,ggvermelho,ggazul):

    G = nx.Graph(graph.edges)

    # G.add_edges_from(E1)
    def node_color(node):
        if node in stp.terminals :
            return 'orange'
        else :
            return 'white'

    def edge_color(v,u):
        if ggvermelho.has_edge(v,u) and ggazul.has_edge(v,u):
            return 'black'
        elif ggvermelho.has_edge(v,u) and not ggazul.has_edge(v,u):
            return 'red'
        elif not ggvermelho.has_edge(v,u) and ggazul.has_edge(v,u):
            return 'steelblue'
        else:
            return 'yellow'

    ed_colors = [edge_color(v,u) for v,u in G.edges() ]

    nd_colors = [node_color(i) for i in G.nodes() ]

    plt.subplot()

    # nx.draw(G, with_labels=True, font_weight='bold')
    # nx.draw_networkx(G,node_size=50,with_labels=False)
    nx.draw_kamada_kawai(G, with_labels=True,node_color=nd_colors,edge_color=ed_colors)
    # nx.draw_kamada_kawai(G, with_labels=False,node_size=50,node_color=nd_colors)

    # pos = nx.kamada_kawai_layout(G)
    # nx.draw_networkx_nodes(G,pos,with_labels=True,node_color=nd_colors)
    # labels = nx.draw_networkx_labels(G, pos=pos)

    # del labels

    plt.show()

    return G

    # G = nx.Graph(stp.graph)

    # plt.subplot()

    # nx.draw_networkx(G,node_size=50,with_labels=False)

    # plt.show()

#######################################################
''
def update_partition(start, end, partition):

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


RED = 0
BOTH = 1
BLUE = 2

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
vertices_done = set()

for adj in GU.adjacent_to(start_node):
    stack.append((start_node, adj))

last_edge_color = BOTH

while stack :
    v, u = stack.pop()

    if sub1.has_edge(v,u) and sub2.has_edge(v,u):
        w = sub1.weight(v,u)
        print('both:  ',v,u)
        last_edge_color = BOTH
        child.add_edge(v,u,weight=w)

    elif sub1.has_edge(v,u) and not sub2.has_edge(v,u):
        print('A:  ',v,u)
        last_edge_color = BLUE
        update_partition(v,u,A_partition)

    elif not sub1.has_edge(v,u) and sub2.has_edge(v,u):
        print('B:  ',v,u)
        last_edge_color = RED
        update_partition(v,u,B_partition)
    else:
        print("ERROR",v,u)

    vertices_done.add(v)

    for t in GU.adjacent_to(u):
        if sub1.has_edge(u,t) and sub2.has_edge(u,t):
            if not t in vertices_done:
                stack.append((u,t))
        elif (last_edge_color is BLUE) and sub2.has_edge(u,t):
            continue
        elif (last_edge_color is RED) and sub1.has_edge(u,t):
            continue
        elif not u in vertices_done: # 
            stack.append((u,t))


print('\n\n\nA:\n')

pp.pprint(A_partition)

print('\n\n\nB:\n')

pp.pprint(B_partition)


