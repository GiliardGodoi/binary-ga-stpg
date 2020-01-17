import pprint as pp
from os import path
#from collections import deque
#from graph.priorityqueue import PriorityQueue

from graph import GraphDictionary
from graph.steiner_heuristics import shortest_path_origin_prim as heuristic
#from graph.steiner_heuristics import prunning_mst


from graph.reader import Reader
from util import gg_rooted_tree, find_tree_path, check_circles_dfs, gg_total_weight


def avaliar_comprimento_caminho(caminho,grafo : GraphDictionary):
    
    soma = 0
    i = 1
    while i < len(caminho):
        v = caminho[i-1]
        u = caminho[i]
        soma += grafo[v][u]
        i += 1
        
    return soma

def adicioar_caminho(caminho, descendente):

    i = 1
    while i < len(caminho):
        descendente[caminho[i-1]] = caminho[i]
        i += 1

def turn_tree_dict_as_graph(rrtree : dict, ggraph : GraphDictionary):
    
    C = GraphDictionary()

    for k, v in rrtree.items():
        if v :
            w = ggraph.weight(k,v)
            C.add_edge(v,k,weight=w)

    return C

if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = GraphDictionary(vertices=stp.nro_nodes, edges=stp.graph)

    sub1, c1 = heuristic(graph, stp.terminals[4], stp.terminals) # 0 ate 16
    sub2, c2 = heuristic(graph, stp.terminals[5], stp.terminals)

    _root = 36

    A = gg_rooted_tree(sub1,_root)
    B = gg_rooted_tree(sub2,_root)
      
    child  = dict()
    child[_root] = None

    for k, v in A.items():
        if v : 
            if k in B and v == B[k]:
                child[k] = v

    vv = A.keys() & B.keys()
    
    for k in child.keys():
       vv.remove(k)
   
    done = set()
    
    while vv :
        
        node = vv.pop()

        _path_A = find_tree_path(A, node, _root)[0]
        _path_B = find_tree_path(B, node, _root)[0]
        
        custo_A = avaliar_comprimento_caminho(_path_A, sub1)
        custo_B = avaliar_comprimento_caminho(_path_B, sub2)
        
        _selecionado = []
        
        if custo_A < custo_B :
            adicioar_caminho(_path_A, child)
            _selecionado = _path_A

        elif custo_B < custo_A :
            adicioar_caminho(_path_B, child)
            _selecionado = _path_B

        else:
            print('caminhos possuem o mesmo custo')
            adicioar_caminho(_path_B, child)
            
        for s in _selecionado:
            if s in vv:
                vv.remove(s)


    CC = turn_tree_dict_as_graph(child,graph)

    c3 = gg_total_weight(CC)
