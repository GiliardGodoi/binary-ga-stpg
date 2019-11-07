import pprint as pp
from os import path

from graph.util.reader import Reader
from graph.graph import GraphDictionary as Graph

diretorio_dados = "datasets"
arquivo_dados = "b01.stp"

arquivo = path.join(diretorio_dados, arquivo_dados)

reader = Reader()

stp = reader.parser(arquivo)

# pp.pprint(stp.graph)

graph = Graph(vertices=stp.nro_nodes,edges=stp.graph)

from heapq import heapify, heappop, heappush

class priority_queue(object):
    
    def __init__(self):
        self.queue = list()
        heapify(self.queue)
        self.index = dict()
 
    def push(self, priority, label):
        if label in self.index:
            self.queue = [(w,l) for w,l in self.queue if l!=label]
            heapify(self.queue)
        heappush(self.queue, (priority, label))
        self.index[label] = priority
 
    def pop(self):
        if self.queue:
            return heappop(self.queue)

    def __contains__(self, label):
        return label in self.index

    def __len__(self):
        return len(self.queue)

def prim(graph, start):
        treepath = {}
        total = 0
        queue = priority_queue()
        queue.push(0 , (start, start))

        while queue:

            weight, (node_start, node_end) = queue.pop()
            if node_end not in treepath:
                treepath[node_end] = node_start
                if weight:
                    total += weight
                for next_node, weight in graph[node_end].items():
                    queue.push(weight , (node_end, next_node))

        return treepath

mst = prim(stp.graph,20)
