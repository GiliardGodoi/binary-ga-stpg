# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 2020

@author: Giliard A Godoi
"""
from os import path

from graph import GraphDictionary, Reader
from graph.steiner_heuristics import shortest_path_with_origin
from util import gg_rooted_tree

class Subset:
    def __init__(self, v1, v2):
        
        self.elements = GraphDictionary()
        self.elements.add_edge(v1,v2)
        
    def add(self,v1, v2):
        self.elements.add_edge(v1,v2)
        
        return True
    
    
    def __len__(self):    
        return len(self.elements)
    
    
    def __iter__(self):
        return self.elements.gen_undirect_edges()
               
        
subsets = dict()
    
def build_disjoint_path(rep, other):
    
        
    if not rep in subsets:
        subsets[rep] = Subset(rep, other)
        
    else :
        subsets[rep].add(rep, other)
    
    # atualizando a representação nas duas partições
    if other in subsets:
            subsets[other] = subsets[rep]
            
    return subsets[rep]


if __name__ == "__main__":
    arquivo = path.join("datasets","b13.stp")

    reader = Reader()

    stp = reader.parser(arquivo)

    graph = GraphDictionary(vertices=stp.nro_nodes, edges=stp.graph)

    s1 = stp.terminals[5]
    subtree_1, cost1 = shortest_path_with_origin(graph, s1, stp.terminals) # 0 ate 16

    s2 = stp.terminals[9]
    subtree_2, cost2 = shortest_path_with_origin(graph, s2, stp.terminals)

    print(cost1, cost2)

    # Primeiro uma das soluções será transformada na sua forma enraizada
    rooted_tree_1 = gg_rooted_tree(subtree_1,s1)

    # Definir o conjunto de arestas que somente pertence a subtree_2
    edges_disjoint_2 = set()

    # uma função auxiliar para padronizar as arestas. Será necessário?
    std_edges = lambda x, y : (min(x,y), max(x,y))
    
    
    # todas as arestas não comum de um dos pais (subtree 2)
    for v, u in subtree_2.gen_undirect_edges():
        if not subtree_1.has_edge(v,u) :
            edges_disjoint_2.add((v,u))
        else:
            print("has edge",v,u)
            
            
    disjoints_paths = dict()
                   
    
    for v, u in edges_disjoint_2:
        
        if subtree_1.has_node(v) and subtree_1.has_node(u):
            disjoints_paths[(v,u)] = Subset(v, u)
            
        elif not subtree_1.has_node(v) and subtree_1.has_node(u):
            build_disjoint_path(v,u)
            
        elif subtree_1.has_node(v) and not subtree_1.has_node(u):
            build_disjoint_path(u,v)
            
        else :
            ss = build_disjoint_path(v,u)
            if not u in subsets:
                subsets[u] = ss
            
            
            
    for i in subsets:
        print(subsets[i].elements)
        print('\n')

