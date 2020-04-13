# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:26:16 2020

@author: Giliard Almeida de Godoi
"""
import os
import random
import statistics
from collections import deque
from operator import attrgetter

from genetic.base import Operator
from genetic.chromosome import BaseChromosome
from genetic.selection import roullete_selection
from genetic.util.datalogger import BaseLogger, SimulationLogger
from graph import Graph, ReaderORLibrary, SteinerTreeProblem
from graph.algorithms import kruskal
from graph.disjointsets import DisjointSets
from graph.steiner_heuristics import shortest_path_with_origin
from graph.util import gg_total_weight, has_cycle
import time


class Chromosome(BaseChromosome):

    def __init__(self, genes):
        super().__init__(genes)

    @property
    def graph(self):
        return self.genes

    @graph.setter
    def graph(self, newgraph):
        self.genes = newgraph


class SimplePartitionCrossover (Operator):
    '''
    Operação de Cruzamento baseado em avaliação das partições.

    1. Identifica as arestas comuns e as arestas não comuns, formando componentes conexas.
    2. Arestas comuns são transmitidas para os descendentes.
    3. Arestas não comuns formam partições (componentes conexas) que serão avaliadas isoladamente.
    4. Para cada partição (passo 3) compara o peso das arestas de cada um dos pais.
    O subconjunto de arestas de menor peso (custo) será transmitido ao descendente.

    Notes:
    Este procedimento não garante que todos os descendentes serão factíveis.
    '''

    def __init__(self, graphinstance : Graph):
        self.graph = graphinstance

    def operation(self, PARENT_A : Chromosome, PARENT_B : Chromosome):
        '''Implementa a operação de crossover'''

        SUBTREE_A = PARENT_A.graph
        SUBTREE_B = PARENT_B.graph

        graph_common = Graph()
        graph_partition = Graph()

        A_vertices = set()
        B_vertices = set()

        for v, u in SUBTREE_A.gen_undirect_edges():
            weight = SUBTREE_A.weight(v,u)
            if SUBTREE_B.has_edge(v,u):
                graph_common.add_edge(v, u, weight=weight)
            else:
                A_vertices.add(v)
                A_vertices.add(u)
                graph_partition.add_edge(v, u, weight=weight)

        for v, u in SUBTREE_B.gen_undirect_edges():
            weight = SUBTREE_B.weight(v, u)
            if not SUBTREE_A.has_edge(v, u):
                B_vertices.add(v)
                B_vertices.add(u)
                graph_partition.add_edge(v, u, weight=weight)

        AandB_vertices = A_vertices.intersection(B_vertices)
        partitions = list()

        while AandB_vertices:
            start = AandB_vertices.pop()
            partition, visited = self.__dfs__(graph_partition,SUBTREE_A, SUBTREE_B, start)

            if partition["A"]["cost"] <= partition["B"]["cost"] :
                partitions.append(partition["A"])
            else :
                partitions.append(partition["B"])

            AandB_vertices = AandB_vertices.difference(visited) # estao em AandB_vertices mas não estão em visited O(n + m)


        for partition in partitions :
            for v, u in partition["edges"]:
                graph_common.add_edge(v, u, weight=self.graph.weight(v, u))

        return Chromosome(graph_common)

    def __dfs__(self, uncommon_graph : Graph, Atree : Graph, Btree : Graph, start):

        vertices_done = set()
        stack = deque([start])

        partition = {
                "A" : {"edges": set(), "cost" : 0 },
                "B" : {"edges": set(), "cost" : 0 }
            }

        while stack:
            node = stack.pop()
            vertices_done.add(node)

            for adj in uncommon_graph.adjacent_to(node):

                if adj not in vertices_done:
                    if Atree.has_edge(node, adj):
                        partition["A"]["edges"].add((node, adj))
                        partition["A"]["cost"] += uncommon_graph.weight(node, adj)
                    elif Btree.has_edge(node, adj):
                        partition["B"]["edges"].add((node, adj))
                        partition["B"]["cost"] += uncommon_graph.weight(node, adj)

                    stack.append(adj)

        return partition, vertices_done


class GeneticAlgorithm:
    '''Define the basic class to define a GA'''

    def __init__(self, STPG : SteinerTreeProblem):
        self.STPG = STPG
        self.graph = Graph(edges=STPG.graph)

        self.crossover_operator = SimplePartitionCrossover(self.graph)
        self.selection_operator = roullete_selection
        self.mutation_operator = lambda item : item ## Null operator

        self.population = list()
        self.population_size = 0
        self.selected_population = list()

        self.tx_crossover = 0.9
        self.tx_mutation = 0.2

        self.tx_newvertice = 0.8

        self.chromosome_legth = 0

        self.best_chromosome = None
        self.last_time_improvement = 0

        self.logger = SimulationLogger()
        self.logger.add_header('best_fitness','iteration', 'cost', 'fitness')
        self.logger.add_header('best_from_round', 'iteration', 'cost', 'fitness')
        self.logger.add_header("evaluation","iteration" , "penalization", "average", "std_deviation")

    def generate_new_individual(self, *args):

        all_edges = args[0]
        random.shuffle(all_edges)
        terminals = set(self.STPG.terminals)

        DS = DisjointSets()
        subgraph = Graph()

        while all_edges:
            v, u, w = all_edges.pop()

            if v not in DS :
                DS.make_set(v)
            if u not in DS:
                DS.make_set(u)

            if DS.find(v) != DS.find(u):
                DS.union(v,u)
                subgraph.add_edge(v, u, weight=w)
                terminals.discard(v)
                terminals.discard(u)

        return Chromosome(subgraph)

        # GRAPH = self.graph
        # terminals = set(self.STPG.terminals)
        # vertices = set(terminals)
        # for v in GRAPH.vertices:
        #     if random.random() < self.tx_newvertice:
        #         vertices.add(v)

        # subgraph = Graph()
        # for v in vertices :
        #     subgraph.add_node(v)
        #     for u in GRAPH.adjacent_to(v):
        #         if u in vertices:
        #             subgraph.add_edge(v, u, weight = GRAPH.weight(v,u))


        # subgraph = kruskal(subgraph)
        # tt = terminals - set(subgraph.vertices)

        # if tt:
        #     # print(':: bad smell :: ')
        #     while tt:
        #         v = tt.pop()
        #         subgraph.add_node(v)
        # return Chromosome(subgraph)

    def generate_population(self, population_size, **kargs):

        self.population_size = population_size
        counter = 0
        newpopulation = list()

        all_edges = [(v, u, self.graph.weight(v,u)) for v, u in self.graph.gen_undirect_edges()]

        while counter < self.population_size:
            newpopulation.append(self.generate_new_individual(all_edges[:]))
            counter += 1

        self.update_population(newpopulation, **kargs)

    def eval_chromosome(self, chromosome : Chromosome):

        if hasattr(chromosome, 'graph') and isinstance(chromosome.graph, Graph):
            graph = chromosome.graph
        elif hasattr(chromosome, 'genes') and isinstance(chromosome.genes, Graph):
            graph = chromosome.genes
        else:
            print(type(chromosome.graph))
            raise TypeError("bad chromosome")

        previous = dict()
        qtd_partitions = 0
        total = {'cost' : 0 } # Gambiarra

        def penality(k): return (k - 1) * 100

        def DFS_visit(start):
            for v in graph.adjacent_to(start):
                if v not in previous:
                    previous[v] = start
                    total['cost'] += graph.weight(v, start)
                    DFS_visit(v)

        for s in graph.vertices:
            if s not in previous:
                previous[s] = None
                qtd_partitions += 1
                DFS_visit(s)

        return total['cost'] + penality(qtd_partitions)

    def evaluate(self, **kargs):
        '''Evaluates the entire population.

        Returns:
            int or float : total population cost
            int or float : maximun cost from the current generation
            float : average cost from the current generation
        '''
        evaluated_costs = list()

        for chromosome in self.population:
            cost = self.eval_chromosome(chromosome)
            chromosome.cost = cost
            evaluated_costs.append(cost)

        return sum(evaluated_costs), max(evaluated_costs), statistics.mean(evaluated_costs)

    def selection(self):
        self.selected_population = self.selection_operator(self.population, self.population_size)

    def recombine(self):

        newpopulation = list()
        population_size = self.population_size
        count = 0

        while count < population_size:
            parent_a, parent_b = random.sample(self.selected_population, k=2)
            ## Como a operação de seleção é executada.
            child = self.crossover_operator(parent_a, parent_b)

            if isinstance(child, (list, tuple)) :
                newpopulation.extend(child)
                count += len(child)
            else :
                newpopulation.append(child)
                count += 1

        self.update_population(newpopulation)

    def mutation(self):
        population_size = self.population_size
        count = 0
        while count < population_size:
            if random.random() < self.tx_mutation:
                self.population[count] = self.mutation_operator(self.population[count])
            count += 1

    def normalize(self, **kargs):

        best_fitness_value = -float("inf")
        best_chromosome = None

        max_cost = max(chromosome.cost for chromosome in self.population)
        population_fitness = list()

        for chromosome in self.population:
            fitness = max_cost - chromosome.cost
            chromosome.fitness = fitness
            population_fitness.append(fitness)

            if chromosome.fitness > best_fitness_value:
                best_fitness_value = chromosome.fitness
                best_chromosome = chromosome

        self.logger.log("evaluation",
            kargs.get("iteration", 0),
            None,
            statistics.mean(population_fitness),
            statistics.stdev(population_fitness))

        self.update_best_chromosome(best_chromosome, **kargs)

    def update_population(self, newpopulation):
        # replace all
        self.population = newpopulation

    def update_best_chromosome(self, chromosome : Chromosome, **kargs):
        if self.best_chromosome is None:
            self.best_chromosome = chromosome
            self.logger.log('best_fitness', kargs.get("iteration", 0), chromosome.cost, chromosome.fitness)
        elif self.best_chromosome.cost > chromosome.cost:
            self.best_chromosome = chromosome
            self.logger.log('best_fitness', kargs.get("iteration", 0), chromosome.cost, chromosome.fitness)

        self.logger.log('best_from_round', kargs.get("iteration", 0), chromosome.cost, chromosome.fitness)

    def sort_population(self):
        '''Sort the population by fitness attribute'''
        self.population.sort(key=attrgetter("fitness"))


## =========================================================================== ##
def print_pop(GA : GeneticAlgorithm):
    for index, p in enumerate(GA.population, start=1):
        print(index, ' -> cost: ', p.cost, ' - fitness: ', p.fitness)

def test_1():

    def __test__():
        filename = os.path.join("datasets", "ORLibrary", "steinb13.txt")

        reader = ReaderORLibrary()

        STPG = reader.parser(filename)

        graph = Graph(vertices=STPG.nro_nodes, edges=STPG.graph)

        ## DETERMINAR DUAS SOLUÇÕES PARCIAIS PELAS HEURISTICAS
        # escolher aleatoriamente um vértice terminal
        s1, s2 = random.sample(STPG.terminals, k=2)

        SUBTREE_A, cost1 = shortest_path_with_origin(graph, s1, STPG.terminals) # 0 ate 16
        SUBTREE_B, cost2 = shortest_path_with_origin(graph, s2, STPG.terminals)

        SPX = SimplePartitionCrossover(graphinstance=graph)

        offspring = SPX.operation(Chromosome(SUBTREE_A), Chromosome(SUBTREE_B))

        offspring_cost = gg_total_weight(offspring.graph)

        print(
                f"Parent A: {cost1}",
                f"Parent B: {cost2}\n"
                f"Offspring: {offspring_cost}"
            )
        print("Has cycle", has_cycle(offspring.graph))

    for i in range(50):
        __test__()
        print("\n========================\n")

def test_2():

    filename = os.path.join("datasets", "ORLibrary", "steinb13.txt")

    reader = ReaderORLibrary()

    STPG = reader.parser(filename)

    GA = GeneticAlgorithm(STPG)
    GA.logger = BaseLogger()

    GA.generate_population(10)
    GA.evaluate()
    GA.normalize()
    GA.sort_population()

    print("\nPOPULATION FITNESS\n")
    print_pop(GA)

    GA.selection()
    GA.recombine()
    GA.evaluate()
    GA.normalize()
    GA.sort_population()

    print("\nPOPULATION FITNESS\n")
    print_pop(GA)

    return GA

def test_3():

    def test_if_steiner_tree(subgraph : Graph):
        return False

    filename = os.path.join("datasets", "ORLibrary", "steinb13.txt")

    reader = ReaderORLibrary()

    STPG = reader.parser(filename)

    GA = GeneticAlgorithm(STPG)
    GA.logger = BaseLogger()

    GA.generate_population(100)
    MAX_GENERATION = 100
    counter = 0

    while counter < MAX_GENERATION:
        print("Iteration: ", counter + 1, end='\r')
        GA.evaluate()
        GA.normalize()
        GA.sort_population()
        GA.selection()
        GA.recombine()
        counter += 1

    GA.evaluate()
    GA.normalize()
    print("\nPOPULATION FITNESS\n")
    print_pop(GA)


    return GA

def simulation(dataset: str, nro_trial = 0, global_optimum = 0):
    '''Run a simulation trial'''

    # Lendo a instância do problema
    reader = ReaderORLibrary()
    STPG = reader.parser(dataset)

    # Definindo o diretório que será destinado os dados
    datafolder = os.path.join("outputdata", "teste", STPG.name)
    if not os.path.exists(datafolder):
        os.makedirs(datafolder) # or mkdir

    ## Parâmetros  comuns a cada execução
    GA = GeneticAlgorithm(STPG)
    GA.tx_crossover = 0.85
    GA.tx_mutation =  0.2
    POPULATION_SIZE = 100
    MAX_GENERATION = 10000
    MAX_LAST_IMPROVEMENT = 500
    GLOBAL_OPTIMUN = global_optimum

    ## Definindo a função com os critérios de parada

    def check_stop_criterions(iteration=0):

        if iteration >= MAX_GENERATION:
            return (False, "max_generation_reached")
        elif GA.last_time_improvement > MAX_LAST_IMPROVEMENT:
            return (False, "stagnation")
        elif GA.best_chromosome.cost == GLOBAL_OPTIMUN :
            return (False, "global_optimum_reached")
        else :
            return (True, "non stop")

    ## Configurando a coleta de informações
    GA.logger.prefix = f'trial_{nro_trial}'
    GA.logger.mainfolder = datafolder

    GA.logger.add_header("simulation",
            "nro_trial",
            "instance_problem",
            "nro_nodes",
            "nro_edges",
            "nro_terminals",
            "tx_crossover",
            "tx_mutation",
            "global_optimum",
            "best_cost",
            "best_fitness",
            "population_size",
            "max_generation",
            "iterations",
            "run_time",
            "max_last_improvement",
            "why_stopped"
            )

    ## =============================================================
    ## EXECUTANDO O ALGORITMO GENÉTICO

    GA.generate_population(POPULATION_SIZE)
    # GA.generate_population(POPULATION_SIZE, opt="MST")
    running = True
    epoch = 0
    timestart = time.time()
    while running:
        GA.evaluate(iteration=epoch)
        GA.normalize(iteration=epoch)
        GA.selection()
        GA.recombine()
        GA.mutation()
        GA.last_time_improvement += 1
        epoch += 1
        running, why_stopped = check_stop_criterions(iteration=epoch)
    time_ends = time.time()

    GA.evaluate(iteration=epoch)

    ## Record general simulation data
    GA.logger.log("simulation",
            nro_trial,
            STPG.name,
            STPG.nro_nodes,
            STPG.nro_edges,
            STPG.nro_terminals,
            GA.tx_crossover,
            GA.tx_mutation,
            GLOBAL_OPTIMUN,
            GA.best_chromosome.cost,
            GA.best_chromosome.fitness,
            POPULATION_SIZE,
            MAX_GENERATION,
            epoch,
            (time_ends - timestart),
            MAX_LAST_IMPROVEMENT,
            why_stopped
            )

    ## Generates the reports
    GA.logger.report()

def main():
    import os
    import time
    from graph import ReaderORLibrary

    NUMBER_OF_TRIALS = 1
    DATASETS = [
        ("steinb1.txt",   82), # 0
        ("steinb2.txt",   83),
        ("steinb3.txt",  138),
        ("steinb4.txt",   59),
        ("steinb5.txt",   61), # 4
        ("steinb6.txt",  122),
        ("steinb7.txt",  111),
        ("steinb8.txt",  104),
        ("steinb9.txt",  220), # 8
        ("steinb10.txt",  86),
        ("steinb11.txt",  88),
        ("steinb12.txt", 174),
        ("steinb13.txt", 165), # 12
        ("steinb14.txt", 235),
        ("steinb15.txt", 318), # 14
        ("steinb16.txt", 127), # 15
        ("steinb17.txt", 131), # 16
        ("steinb18.txt", 218), # 17
    ]

    # Prestar atenção ao tempo de execução. Isso pode demorar bastante.
    for filename, global_optimum in DATASETS[:1]:
        dataset = os.path.join("datasets","ORLibrary", filename)
        if os.path.exists(dataset):
            print(dataset, global_optimum)
            print("Executing trial for : ", filename, end="\r")
            for nro in range(1, NUMBER_OF_TRIALS + 1):
                print("Executing trial for : ", filename," Trial nro: ", nro, end="\r")
                simulation(dataset, nro_trial=nro, global_optimum=global_optimum)


if __name__ == "__main__":
    import os
    # test_2()
    main()
