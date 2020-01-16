import pprint as pp
from os import path
from collections import deque

from graph.graph import GraphDictionary
from graph.steiner_heuristics import shortest_path_origin_prim
#from graph.steiner_heuristics import prunning_mst
from util import gg_union,gg_total_weight
from graph.mst import prim

from graph.reader import Reader

'''
    O cruzamento é definido da seguinte forma: é gerado o grafo GU 
    que é a união de duas soluções parciais sub1 e sub2.

    Então é definida a MST de GU utilizando o procedimento de Prim.
    A partir dessa MST, todos os vértices folhas (grau igual a um) 
    que não sejam vértices terminais são eliminados. Este procedimento é repetido
    iterativamente até que restem somente vértices folhas que sejam terminais.

    Entretanto este procedimento pode não gerar uma solução de custo menor em relação
    as soluções parciais inicialmente consideradas.
    O procedimento de Prim não sabe desconsiderar vértices não-obrigatórios da formação da MST.
    Assim, longos caminhos com apenas vértices não obrigatórios são considerados ao invés de 
    atalhos existentes entre vértices terminais não são considerados.
'''


if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = GraphDictionary(vertices=stp.nro_nodes, edges=stp.graph)

    sub1, c1 = shortest_path_origin_prim(graph, stp.terminals[8], stp.terminals)
    sub2, c2 = shortest_path_origin_prim(graph, stp.terminals[1], stp.terminals)
    
    GU = gg_union(sub1, sub2)
    
    tree, cu = prim(GU,stp.terminals[1])
    
    GU_prim_tree = GraphDictionary()
    
    for k, v in tree.items():
        if k != v :
            w = GU.weight(k,v)
            GU_prim_tree.add_edge(k, v,weight=w)
            
            
    terminals = set(stp.terminals)
    delete_nodes = deque()
    
    for v in GU_prim_tree.vertices:
        if GU_prim_tree.degree(v) == 1 and v not in terminals:
            delete_nodes.append(v)
            
            
    while delete_nodes:
        
        print(delete_nodes)
        
        v = delete_nodes.pop()
        
        for u in GU_prim_tree.adjacent_to(v):
            if GU_prim_tree.degree(u) == 2 and u not in terminals:
                delete_nodes.append(u)
        
        GU_prim_tree.remove_node(v)
        
    
    custo_arvore_gerada = gg_total_weight(GU_prim_tree)
    
    print(c1)
    print(c2)
    print(custo_arvore_gerada)
    
    
