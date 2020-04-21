import os
import random
import statistics
import time
from collections import deque

from ga_simplestpartition import SimplePartitionCrossover
from genetic.base import BaseGA, Operator
from genetic.chromosome import BinaryChromosome, TreeBasedChromosome
from genetic.crossover import crossover_2points
from genetic.datalogger import DataLogger
from genetic.mutation import mutation_flipbit
from genetic.selection import roullete_selection
from graph import Graph, ReaderORLibrary
from graph.disjointsets import DisjointSets
from graph.priorityqueue import PriorityQueue
from util import (convert_binary2treegraph, convert_treegraph2binary,
                  evaluate_binary, evaluate_treegraph,
                  random_binary_chromosome, random_treegraph_chromosome)

class HybridGeneticAlgorithm(BaseGA):

    def __init__(self, STPG, **kargs):
        super().__init__(**kargs)
        self.STPG = STPG
        self.GRAPH = Graph(edges=STPG.graph)
        self.terminals = set(STPG.terminals)
        self.nro_vertices = STPG.nro_nodes
        self.chromosome_length = STPG.nro_nodes - STPG.nro_terminals

    def generate_new_individual(self, **kargs):
        return random_binary_chromosome(self.chromosome_length)

    def eval_chromosome(self, chromosome):

        def penality(nro_partitions):
            return (nro_partitions - 1) * 100

        if type(chromosome) is BinaryChromosome:
            return evaluate_binary(chromosome,self.GRAPH, self.terminals, self.nro_vertices, penality)

        elif type(chromosome) is TreeBasedChromosome:
            return evaluate_treegraph(chromosome, penality)
        else:
            raise TypeError("chromosome cannot be evaluated")

    def mutation(self, active=False):
        if active:
            super().mutation()

class Simulation:

    def __init__(self, name='teste', **kwargs):
        self.name = name

        self.STPG = None
        self.GRAPH = None

        self.CONTROL_FLAG = True
        self.MAX_ITERATIONS = 10000
        self.POPULATION_SIZE = 100

    def setUp(self, dataset='', **kargs):
        print("executing setUP...")

        self.MAX_ITERATIONS = kargs.get("max_iterations", self.MAX_ITERATIONS)

        self.setup_dataset(dataset)
        self.setup_ga()

    def setup_dataset(self, dataset):
        print(f"setting instance problem from dataset : {dataset}")
        filename = os.path.join("datasets", "ORLibrary", dataset)
        reader = ReaderORLibrary()
        self.STPG = reader.parser(filename)
        self.GRAPH = Graph(edges=self.STPG.graph)

    def setup_ga(self):
        print("setting GA configurations ...")
        self.GA = HybridGeneticAlgorithm(self.STPG)
        self.CONTROL_FLAG = True # if True --> BinaryRepresentation

        self.GA.tx_crossover = 0.85
        self.GA.tx_mutation =  0.4
        self.GA.population_size = self.POPULATION_SIZE


        self.gpx_strategy = SimplePartitionCrossover(self.GRAPH)

        self.GA.crossover_operator = crossover_2points if self.CONTROL_FLAG else self.gpx_strategy
        self.GA.selection_operator = roullete_selection
        self.GA.mutation_operator = mutation_flipbit

        self.GA.logger = DataLogger()
        self.GA.logger.mainfolder = self.name
        self.GA.logger.register("simulation", "csv",
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

    def check_stop_criterion(self, **kargs):
        iteration = kargs.get("iteration", self.MAX_ITERATIONS)
        return iteration < self.MAX_ITERATIONS

    def run(self, filename, **kargs):
        self.setUp(dataset=filename)

        print("Starting GA execution ...")
        GA = self.GA
        print("generate_population ...")
        GA.generate_population()
        iteration = 0
        start_at = time.time()
        while self.check_stop_criterion(iteration=iteration):
            print("                                                 Iteration:  ", iteration, end="\r")
            GA.evaluate()
            GA.normalize(iteration=iteration)
            GA.sort_population()
            GA.selection()
            GA.recombine()
            GA.mutation(active=self.CONTROL_FLAG)
            iteration += 1
            if (iteration % 200) == 0:
                # print("not changing!")
                self.change_the_game()

        GA.evaluate()
        GA.normalize(iteration=iteration)

        ends_at = time.time()

        data = {
            "nro_trial" : kargs.get("trial", 1),
            "global_optimum" : 0,
            "iterations" : iteration,
            "run_time" : (ends_at - start_at),
            "max_last_improvement" : self.GA.last_time_improvement,
            "why_stopped" : "max_iteration_reached"
        }
        self.finish_it(**data)

    def run_trials(self, nro_trials):
        for trial in range(1, nro_trials+1):
            self.run(trial=trial)

    def finish_it(self, **kargs):

        self.GA.logger.log("simulation",
            kargs.get("nro_trial", 0),
            self.STPG.name,
            self.STPG.nro_nodes,
            self.STPG.nro_edges,
            self.STPG.nro_terminals,
            self.GA.tx_crossover,
            self.GA.tx_mutation,
            kargs.get("global_optimum", None),
            self.GA.best_chromosome.cost,
            self.GA.best_chromosome.fitness,
            self.POPULATION_SIZE,
            self.MAX_ITERATIONS,
            kargs.get("iterations", 0),
            kargs.get("run_time", 0),
            kargs.get("max_last_improvement", 0),
            kargs.get("why_stopped", "not_provided")
            )

        ## Generates the reports
        self.GA.logger.report()

    def change_the_game(self):
        print("changing ...                                         ")
        # 1st thing to do: change the control flag
        self.CONTROL_FLAG = not self.CONTROL_FLAG

        # 2nd one is change the crossover strategy
        self.GA.crossover_operator = crossover_2points if self.CONTROL_FLAG else self.gpx_strategy
        print(self.GA.crossover_operator.__class__.__name__) ##

        # 3rd one is convert the chromosome representation
        newpopulation = list()
        for chromosome in self.GA.population:
            if self.CONTROL_FLAG:
                newpopulation.append(convert_treegraph2binary(chromosome, self.GA.terminals, self.GA.nro_vertices))
            else:
                newpopulation.append(convert_binary2treegraph(chromosome,self.GA.GRAPH, self.GA.terminals, self.GA.nro_vertices))

        # 4th one is update the population
        self.GA.population = newpopulation


def test_1():
    simulation = Simulation()

    simulation.MAX_ITERATIONS = 1000
    simulation.POPULATION_SIZE = 10

    simulation.run("steinb13.txt")

if __name__ == "__main__":
    test_1()
