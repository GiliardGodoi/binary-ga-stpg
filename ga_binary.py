# -*- coding: utf-8 -*-
import random
import time
from collections import defaultdict
from operator import attrgetter

import statistics

from datalogger import SimulationLogger, Logger
from graph import Graph, ReaderORLibrary, SteinerTreeProblem
from graph.algorithms import prim

## SELECTIONS METHODS

def tournament_selection(population):
    selected = list()
    pool_size = len(population)
    count = 0

    while count < pool_size:
        c1, c2 = random.sample(population, k=2)
        if c1.fitness < c2.fitness:
            selected.append(c1)
        else:
            selected.append(c2)

        count += 1

    return selected


def roullete_selection(population, pool_size):
    # pool_size = len(population)
    fitnesses = [c.fitness for c in population if c.normalized]

    # Return a k sized list of population elements chosen with replacement
    selected = random.choices(population, weights=fitnesses, k=pool_size)

    return selected

##  CROSSOVER METHODS FOR BINARY CHROMOSOMES

def crossover_2points(A_parent, B_parent):
    length = len(A_parent)
    points = random.sample(range(0, length), k=2)
    points.sort()
    p1, p2 = points

    crossing = lambda genex, geney : genex[:p1] + geney[p1:p2] + genex[p2:]

    offspring_A = Chromosome(crossing(A_parent.genes, B_parent.genes))
    offspring_B = Chromosome(crossing(B_parent.genes, A_parent.genes))

    return offspring_A, offspring_B


def crossover_1points(A_parent, B_parent):
    length = len(A_parent)
    point = random.choice(range(0,length))

    crossing = lambda genex, geney : genex[:point] + geney[point:]

    offspring_A = Chromosome(crossing(A_parent.genes, B_parent.genes))
    offspring_B = Chromosome(crossing(B_parent.genes, A_parent.genes))

    return offspring_A, offspring_B

## MUTATION METHODS

def mutation_flipbit(chromosome):
    '''Flip exactly one bit from the chromosome genes'''

    flipbit = lambda x : '1' if x == '0' else '0'

    index = random.randrange(0, len(chromosome))
    genes = chromosome.genes
    genes = genes[:index] + flipbit(genes[index]) + genes[(index + 1):]

    return Chromosome(genes)


## NORMALIZATION METHODS


## CHROMOSOME CLASS

class Chromosome(object):

    def __init__(self, genes):
        self.genes = genes
        self.__cost = 0
        self.__fitness = 0
        self.normalized = False

    @property
    def cost(self):
        return self.__cost

    @cost.setter
    def cost(self, value):
        self.__cost = value
        self.__fitness = value
        self.normalized = False

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value
        self.normalized = True

    def __len__(self):
        return len(self.genes)

    def __str__(self):
        return str(self.genes)

    def __repr__(self):
        return self.__str__()


class GeneticAlgorithm(object):
    '''
    Basic implementation of mainly fuctions for a GA
    with binary chromosome representation.
    '''
    def __init__(self, STPG : SteinerTreeProblem):
        self.graph = Graph(edges=STPG.graph)
        self.terminals = set(STPG.terminals)
        self.nro_vertices = STPG.nro_nodes

        self.population = list()
        self.popultaion_size = 0

        self.tx_mutation = 0.2
        self.tx_crossover = 0.85

        self.best_chromosome = None
        self.last_time_improvement = 0

        # Quantidade de vértices não obrigatórios.
        self.chromosome_length = STPG.nro_nodes - STPG.nro_terminals

        ## MAPPING NODES DICTIONARY
        # O cromossomo representa unicamente os vértices não-obrigatórios de um grafo.
        # por isso uma estrutura de dicionário é utilizado para mapear o indice do gene
        # no cromossomo para o vértice que aposição representa.
        # O dicionário possui a forma:
        #       { <indice posição do vetor> : <vértice representado> }
        self.map_nodes = dict()
        index = 0
        for node in range(1, (STPG.nro_nodes+1)):
            if not node in self.terminals:
                self.map_nodes[index] = node
                index += 1

        self.logger = SimulationLogger()
        # self.logger.add_header('crossover','A_parent', 'B_parent')
        self.logger.add_header('best_fitness','iteration', 'cost', 'fitness')
        self.logger.add_header('best_from_round', 'iteration', 'cost', 'fitness')
        self.logger.add_header("evaluation","iteration" , "penalization", "average", "std_deviation")

    def __mapper(self, index):
        return self.map_nodes[index]

    def eval_chromosome(self, chromosome : Chromosome):
        ''' Evaluete chromosome's fitness.

        Parameter
            chromosome : Chromosome class

        Return
            int : fitness value. Though it use MST to compute fitness, it might added to a penalty value.
            bool : indicates if it was added a penalization

        Notes
            1. Define o subgrafo induzido pela união dos vértices presentes no cromossomo
                (vértices não-obrigatórios) e vértices terminais - uma aresta pertence ao
                subgrafo induzido se os seus vértices extremos pertencem a esse subconjunto.

            2. Calcula a MST de cada componente conexa k ( com k >= 1) obtendo um custo total;

            3. Define uma penalidade linearmente dependente de k

            4. Transforma em um problema de maximização subtraindo de cada fitness o maior valor de fitness observado na população.
        '''
        subgraph, vertices = self.decode_chromosome(chromosome)

        penalize = lambda k : (k-1) * 100
        total_cost = 0
        nro_partition = 0

        while vertices:
            v = vertices.pop()
            tree, cost = prim(subgraph, v)
            total_cost += cost
            nro_partition += 1

            for w in tree.keys():
                vertices.discard(w)

        penalization = penalize(nro_partition)
        total_cost = total_cost + penalization

        return total_cost, penalization > 0

    def generate_random_individual(self):
        '''Generates a single chromosome'''
        length = self.chromosome_length

        genes = ''.join(random.choices(['0', '1'], k=length))

        return Chromosome(genes)

    def generate_population(self, population_size, opt=None):

        assert population_size > 0, "Population must have more than 0 individuals"
        assert (population_size % 2) == 0, "Population size must be a even number"

        self.population_size = population_size

        population = list()
        count = 0

        if opt and opt == "MST" :
            genes = ''.join(['1'] * self.chromosome_length)
            population.append(Chromosome(genes))
            count += 1
            print("MST seeded...")
        elif opt and opt == "TRIM" :
            print("TRIM seeded")

        for _ in range(count, population_size):
            population.append(self.generate_random_individual())

        self.population = population

    def evaluate(self, **kargs):

        population_cost = 0
        max_cost = 0

        best_fitness = - float("inf") # minus infinity
        bfn_chromosome = None # Best For Now

        count_penalized = 0

        for chromosome in self.population:
            cost, penalized = self.eval_chromosome(chromosome)
            if penalized: count_penalized += 1

            chromosome.cost = cost
            population_cost += cost

            if cost > max_cost:
                max_cost = cost

        # O procedimento abaixo pode ser considerando uma operação de
        # NORMALIZAÇÃO
        # foi mantido no método de avaliação  porque:
        # 1. O maior custo já foi determinado; não precisaria determinar novamente
        # no método 'self.normalize'
        # 2. No artigo ele descrito como um procedimento do fitness.
        # Então resolvi ser consistente com o artigo original.
        # Mas esse procedimento poderia ser implementado no método de normalização
        fitness_values = list()
        for chromosome in self.population:
            chromosome.fitness = max_cost - chromosome.cost
            fitness_values.append(chromosome.fitness)

            if chromosome.fitness > best_fitness:
                best_fitness = chromosome.fitness
                bfn_chromosome = chromosome

        self.update_best_chromossome(bfn_chromosome,**kargs)
        iteration = kargs.get("iteration", 0)
        avg_fitness, std_fitness = statistics.mean(fitness_values), statistics.stdev(fitness_values)
        self.logger.log("evaluation", iteration, count_penalized, avg_fitness, std_fitness)

        return population_cost

    def selection(self):
        self.population = roullete_selection(self.population, self.population_size)

    def recombine(self):

        newpopulation = list()
        population_size = self.population_size
        count = 0

        while count < population_size:
            parent_a, parent_b = random.sample(self.population, k=2)
            child_a, child_b = crossover_2points(parent_a, parent_b)

            newpopulation.append(child_a)
            newpopulation.append(child_b)
            count += 2
        ## Após a aplicação da operação de cruzamento o chromossomo não tem o seu fitness
        ## avaliado imediatamente. Algumas estratégias podem ser utilizadas...

        # Posso inserir todos os novos individuos em um pool e na operação de seleção
        # na função <selection> selecionar apenas o número original de individuos.
        # Assim se a minha população inicial é de 100 individuos, a operação de recombinação
        # irá gerar o dobro de individuos (200) e na fase de seleção eu seleciono apenas 100, de
        # de acordo com a estratégia de seleção utilizada, no caso seleção por roleta.

        # Ou podemos em <update_population> selecionar os N primeiros melhores chromossomos.
        # onde N é o número de individuos da nossa população.
        # mas essa estratégia requer um passo de avaliação extra dos cromossomos.

        # ou eu posso simplesmente adicionar 2 ao contador <count> como está sendo

        self.updata_population(newpopulation)

    def mutation(self):
        population_size = self.population_size
        count = 0

        while count < population_size:
            if random.random() < self.tx_mutation:
                self.population[count] = mutation_flipbit(self.population[count])
            count += 1

    def normalize(self):
        pass

    def elitism(self):
        pass

    def update_best_chromossome(self, chromosome, **kargs):
        '''It traces the best solution found out so far'''
        iteration = kargs.get("iteration", 0)

        if not self.best_chromosome:
            self.best_chromosome = chromosome
            self.logger.log('best_fitness', iteration, chromosome.cost, chromosome.fitness)
            self.last_time_improvement = 0

        elif chromosome.cost < self.best_chromosome.cost:
            self.best_chromosome = chromosome
            self.logger.log('best_fitness', iteration, chromosome.cost, chromosome.fitness)
            self.last_time_improvement = 0

        self.logger.log('best_from_round', iteration, chromosome.cost, chromosome.fitness)

    def updata_population(self, new_population):
        '''It's execute the population replace strategy'''

        assert len(self.population) == len(new_population), "It is not the same size"
        ## Replace all
        self.population = new_population

    def sort_population(self):
        '''Sort the population by fitness attribute'''
        self.population.sort(key=attrgetter("fitness"))

    def decode_chromosome(self, chromosome : Chromosome):
        '''Define the subgraph from a particular chromosome representation.

        Parameter:
            chromosome : Chromosome

        Returns
            Graph: represents the subtree induced by the vertices presented in the chromosome.
            set : all the vertices labels in a set - just to save effort.

        '''
        vertices = set(self.terminals)
        subgraph = Graph()

        for index, gene in enumerate(chromosome.genes):
            if gene == '1':
                node = self.__mapper(index)
                vertices.add(node)

        for v in vertices:
            subgraph.add_node(v)
            for u in self.graph.adjacent_to(v):
                if u in vertices :
                    w = self.graph.weight(v, u)
                    subgraph.add_edge(v, u, weight=w)

        return subgraph, vertices

    def encode_chromosome(self, subgraph: Graph):
        '''Enconde a subgraph using the chromosome representation

        Notes:
        It's dont check if subgraph is actually a subgraph from self.graph
        '''
        pass


def run_trial(dataset: str, trial = 0, global_optimum = 0):

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
    GA.logger.prefix = f'trial_{trial}'
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
            trial,
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


if __name__ == "__main__":
    import os
    import time
    from tqdm import tqdm
    from graph import ReaderORLibrary

    NUMBER_OF_TRIALS = 30
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
    for filename, global_optimum in DATASETS:
        dataset = os.path.join("datasets","ORLibrary", filename)
        if os.path.exists(dataset):
            print(dataset, global_optimum)
            print("Executing trial for : ", filename, end="\r")
            for trial in range(1, NUMBER_OF_TRIALS + 1):
                print("Executing trial for : ", filename," Trial nro: ", trial, end="\r")
                run_trial(dataset, trial, global_optimum=global_optimum)



    # reader = ReaderORLibrary()
    # STPG = reader.parser(dataset)

    # GA = GeneticAlgorithm(STPG)

    # GA.generate_population(10)

    # time_starts = time.time()
    # for iteration in tqdm(range(0,10000)):
    #     # print("Iteration: ", (iteration + 1), end="\r")
    #     GA.evaluate()
    #     GA.selection()
    #     GA.recombine() # problema identificado aqui
    #     GA.mutation()

    # time_ends = time.time()

    # GA.evaluate()

    # GA.logger.report()


    # print("Total run time: ", (time_ends - time_starts))
    # subgraph, _ = GA.decode_chromosome(GA.best_chromosome)

    # # has_cycle, _ = check_cycle_dfs(subtree, STPG.terminals[0])
    # dtree, cost = prim(subgraph, STPG.terminals[0])
    # fitness = GA.eval_chromosome(GA.best_chromosome)
