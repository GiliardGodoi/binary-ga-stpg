"""
    Parametros da simulação

    MAX_GENERATION   : número máximo de iterações
    POPULATION_SIZE  : quantidade de individuos da simulação
    INSTANCE_PROBLEM : qual a instância do problema que vamos rodar
    MAX_TRIALS       : quantidade de repetições que vamos fazer daquela simulaçõa.

    Para definir uma simulação é preciso

        - Definir os critérios de parada.
        - Definir os loggers e o diretório onde os dados da simulação serão persistidos.
"""
from datetime import datetime
from os import path
import time

import config
from genetic.binary import random_binary
from genetic.binary.crossover import crossover_2points, crossover_uniform
from genetic.binary.mutate import flip_nbit
from genetic.binary.selection import roullete
from genetic.condition import StopConditionReached, condition
from genetic.datalogger import DataLogger
from genetic.population import BasePopulation as Population
from logger import get_logger, filename_date
from tools import evaluate_binary
from util import instance_problem

syslog = get_logger(__name__, file_dst=filename_date("simulation"))

class SimulationData:

    def __init__(self):
        self.name = None
        self.trial = None
        self.tx_crossover = 1
        self.tx_mutation = None
        self.globaloptimum = None
        self.max_generation = None
        self.run_time = None
        self.stoppedby = None

def pipe_crossover2points(population, **kwargs):
    # ------------INICÍO DO GA-------------------------------------------
    while condition.check(population):
        population.select(selector_func=roullete)
        population.recombine(mate_func=crossover_2points)
        population.mutation(mutate_func=flip_nbit, tx_mutation=0.2)
        population.evaluate()
        population.normalize()
        population.generation += 1
    population.evaluate()
    # ------------------------------------------------------------------

    return population


def pipe_crossoveruniform(population, **kwargs):
    # ------------INICÍO DO GA-------------------------------------------
    while condition.check(population):
        population.select(selector_func=roullete)
        population.recombine(mate_func=crossover_uniform)
        population.mutation(mutate_func=flip_nbit)
        population.evaluate()
        population.normalize()
        population.generation += 1
    population.evaluate()
    # ------------------------------------------------------------------
    return population


def simulation(population_size: int = 100,
                n_iterations: int = 10_000,
                n_trials : int = 1,
                improvement_interval : int = 500,
                stpg_filename : str = None,
                best_known_solution: int = None,
                evol_func : callable = None,
                simulation_name : str = None
            ):

    syslog.info("INIT SIMULATION")
    syslog.info(f"  params   simulation_name       {simulation_name}")
    syslog.info(f"  params   population_size       {population_size}")
    syslog.info(f"  params   n_iterations          {n_iterations}")
    syslog.info(f"  params   improvement_interval  {improvement_interval}")
    syslog.info(f"  params   stpg_filename         {stpg_filename}")
    syslog.info(f"  params   best_known_solution   {best_known_solution}")
    syslog.info(f"  params   evol_func         {evol_func.__name__}")


    simdata = SimulationData()
    simdata.name = simulation_name
    simdata.max_generation = n_iterations
    simdata.globaloptimum = best_known_solution

    @condition
    def max_generation(population : Population):
        return population.generation < n_iterations

    @condition.params(active=(best_known_solution is not None))
    def best_known_reached(population : Population):
        return not population.best_chromosome.cost == best_known_solution

    @condition.params(active=(best_known_solution is not None))
    def bestcost_lessthan(population: Population):
        return best_known_solution <= population.best_chromosome.cost

    @condition.params(active=(improvement_interval is not None))
    def last_time_improvement(population : Population):
        lstimprovement = population.best_chromosome.last_improvement
        generation = population.generation
        return (generation - lstimprovement) <= improvement_interval

    syslog.info("CHECKING CONDITIONS")
    syslog.info(f"   {max_generation}")
    syslog.info(f"   {best_known_reached}")
    syslog.info(f"   {bestcost_lessthan}")
    syslog.info(f"   {last_time_improvement}")


    STPG = instance_problem("datasets", "ORLibrary", stpg_filename)
    chromosome_length = STPG.nro_nodes - STPG.nro_terminals

    syslog.info(f"  STPG :: {STPG.name},{STPG.nro_nodes},{STPG.nro_edges},{STPG.nro_terminals},{STPG.file_name}")
    syslog.info(f"  params chromosome_length    {chromosome_length}")

    def eval_for(STPG):
        def wrapps(chromosome):
            return evaluate_binary(chromosome, STPG.graph, STPG.terminals, STPG.nro_nodes, lambda k : (k-1) * 100 )

        return wrapps

    def run(trial, **kwargs):

        simdata.trial = trial

        if not condition.conditions:
            raise RuntimeError("Stop conditions weren't defined")

        folder = path.join(config.output_folder, simulation_name, STPG.name)
        logger = DataLogger(prefix=f"trial_{trial}", outputfolder=folder)


        population = Population(chromosomes=[random_binary(chromosome_length) for _ in range(population_size)],
                             evaluation_func=eval_for(STPG))
        population.datacolector = logger
        population.evaluate().normalize()

        start = time.time()
        try:
            evol_func(population)

        except StopConditionReached as msg:
            simdata.stoppedby = str(msg)
            syslog.info(f"  stopped by {simdata.stoppedby}")

        except Exception as error:
            syslog.exception("An error occured")
            raise error

        finally:
            simdata.run_time = time.time() - start
            if not simdata.stoppedby:
                simdata.stoppedby = 'unkonw'

            logger.log_simulation(population, STPG, simdata)
            syslog.info('init report: writting simulation data')
            logger.report()
            syslog.info('finish reporting data')

        # with the return statement is inside the finally block
        # the for loop keeps still running
        # the KeyInterrupedError is missing
        return population


    with max_generation, \
        last_time_improvement, \
        best_known_reached, \
        bestcost_lessthan :
        for trial in range(n_trials):
            trial +=1
            syslog.info(f">> running trial {trial}/{n_trials} for {simulation_name}")
            run(trial)
        else:
            syslog.info('>> executed loop just fine')

    syslog.info("  FINISHI SIMULATION")

if __name__ == "__main__":

    simulation(population_size= 100,
                n_iterations= 500,
                n_trials = 6,
                improvement_interval = 500,
                stpg_filename = "steinb15.txt",
                best_known_solution = 318,
                evol_func = pipe_crossoveruniform,
                simulation_name = "test_crossoveruniform"
            )


    simulation(population_size= 100,
                n_iterations= 5_000,
                n_trials = 6,
                improvement_interval = 500,
                stpg_filename = "steinb15.txt",
                best_known_solution = 318,
                evol_func = pipe_crossover2points,
                simulation_name = "test_crossover_2points"
            )
