# import os
# from graph import Reader
# from graph import GraphDictionary

# diretorio_dados = "datasets"
# arquivo_dados = "b01.stp"
# arquivo = os.path.join(diretorio_dados, arquivo_dados)

# reader = Reader()

# stp = reader.parser(arquivo)
# graph = GraphDictionary(vertices=stp.nro_nodes,edges=stp.graph)


from collections import deque

graph = {'A': ['B', 'C'],
             'B': ['C', 'D'],
             'C': ['D'],
             'D': ['C'],
             'E': ['F'],
             'F': ['C']}


 # Code by Eryk Kopczyński
def find_shortest_path(graph, start, end):
    dist = {start: [start]}
    q = deque(start)
    while len(q):
        at = q.popleft()
        for next in graph[at]:
            if next not in dist:
                dist[next] = [dist[at], next]
                q.append(next)
    return dist