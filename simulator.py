import logging
import os
import time
from operator import attrgetter

import config
from genetic.chromosome import BinaryChromosome, TreeBasedChromosome
from genetic.datalogger import DataLogger
from graph import Graph, ReaderORLibrary
from tools import convert_binary2treegraph

## =====================================================================
##         Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s : %(name)s : %(message)s")

file_handler = logging.FileHandler(os.path.join(config.log_folder, "simulator.log"))
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

## =====================================================================

class ParameterNotFoundError(AttributeError):
    '''Parameter not defined'''

class SimulatorGA():

    def __init__(self, name, params, *args, **kwargs):

        self.simulation_name = name
        self.params = params

        self.STPG = None
        self.GA = None

    def setup_dataset(self, filename, **kwargs):
        logger.info(f"Setting instance problem from dataset {filename}")
        filename = os.path.join(config.datasets_folder, filename)
        reader = ReaderORLibrary()
        self.STPG = reader.parser(filename)
        # self.GRAPH = Graph(edges=self.STPG.graph)

    def setup_ga(self, *args, **kwargs):
        logger.info("Setting GA configuration")

        GAClass = self.GAclass
        ## inicialize properly GA class
        self.GA = GAClass(self.STPG, self.params)

        self.setup_datalogger(**kwargs)

    def setup_datalogger(self, **kwargs):
        folder = os.path.join(config.output_folder, self.simulation_name, self.STPG.name)
        trial = str(kwargs.get("nro_trial", ''))
        self.GA.logger = DataLogger(prefix=f"trial_{trial}", outputfolder=folder)

    def set_gaclass(self, GAclass):
        self.GAclass = GAclass

    def set_stop_condition(self):
        pass

    def check_parameters(self, GA, add_names = []):

        check_for = [
            "tx_crossover",
            "tx_mutation",
            "population_size",
            "max_generation",
            "last_time_improvement",
            "best_chromosome",
            "crossover_operator",
            "selection_operator",
            "mutation_operator"
        ]

        if add_names and isinstance(add_names, list) :
            check_for.extend(add_names)

        for attr in check_for :
            if not hasattr(self, attr):
                raise ParameterNotFoundError(f"Attribute Not Found {attr}")

    def stop_condition(self, iteration=0, **kwargs):
        return iteration < self.params["max_generation"], "always"

    def run(self, *args, **kwargs):

        optimum = kwargs.get("global_optimum", 0)
        iteration = 0
        stop_condition = self.stop_condition

        logger.info("starting GA execution")
        GA = self.GA
        GA.generate_population()

        running, why_stopped = stop_condition(iteration=iteration, global_optimum=optimum)

        start_at = time.time()
        while running:
            print("                                                 Iteration:  ", iteration, end="\r")
            GA.evaluate(iteration=iteration)
            GA.sort_population()
            GA.selection()
            GA.recombine()
            GA.mutation()
            iteration += 1
            GA.check_it(iteration=iteration) ## GA.last_time_improvement += 1
            running, why_stopped = stop_condition(iteration=iteration, global_optimum=optimum)

        GA.evaluate()
        GA.normalize(iteration=iteration)

        ends_at = time.time()

        output = {
            "nro_trial" : kwargs.get("trial", None),
            "global_optimum" : kwargs.get("global_optimum", None),
            "iterations" : iteration,
            "run_time" : (ends_at - start_at),
            "max_last_improvement" : self.GA.last_time_improvement,
            "why_stopped" : why_stopped
        }

        self.report_it(**output)

    def run_multiply_trials(self, DATASETS, NUMBER_OF_TRIALS, *args, **kwargs):
        '''Run multiplus trials of the experiment'''
        for filename, global_optimum in DATASETS:
            logger.info(f"Executing    {filename}")
            self.setup_dataset(filename)
            for trial in range(1, NUMBER_OF_TRIALS + 1):
                self.setup_ga()
                logger.info(f"Executing     {filename}    Trial    {trial}")
                self.run(trial=trial)

    def report_it(self, *args, **kwargs):

        self.GA.logger.log("simulation",
            kwargs.get("nro_trial", 0),
            self.STPG.name,
            self.STPG.nro_nodes,
            self.STPG.nro_edges,
            self.STPG.nro_terminals,
            self.GA.tx_crossover,
            self.GA.tx_mutation,
            kwargs.get("global_optimum", None),
            self.GA.best_chromosome.cost,
            self.GA.best_chromosome.fitness,
            self.GA.population_size,
            self.params["max_generation"],
            kwargs.get("iterations", 0),
            kwargs.get("run_time", 0),
            kwargs.get("max_last_improvement", 0),
            kwargs.get("why_stopped", "not_provided")
            )

        ## Generates the csv data files
        self.GA.logger.report()

    def report_solution_found(self):
        self.GA.logger.register("solution", "json")

        if type(self.GA.best_chromosome) is BinaryChromosome:
            tree_chromo = convert_binary2treegraph(self.GA.best_chromosome, self.GRAPH, self.STPG.terminals, self.STPG.nro_nodes)

            self.GA.logger.log("solution", tree_chromo.genes.edges)

        elif type(self.GA.best_chromosome) is TreeBasedChromosome:
            self.GA.logger.log("solution", self.GA.best_chromosome.genes.edges)
