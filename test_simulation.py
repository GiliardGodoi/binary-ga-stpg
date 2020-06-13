
from genetic.binary import random_binary
from genetic.population import BasePopulation as Population
from genetic.binary.crossover import crossover_2points, crossover_uniform
from genetic.binary.mutate import flip_nbit
from genetic.binary.selection import roullete
from genetic.condition import condition
from genetic.simulator import simulation
from util import instance_problem
from tools import evaluate_binary
import time

MAX_GENERATION = 10

POPULATION_SIZE = 100

STPG = instance_problem("datasets", "ORLibrary","steinb15.txt")

chromosome_length = STPG.nro_nodes - STPG.nro_terminals

@condition
def stopby_maxgeneration(population : "Population"):
    return population.generation < MAX_GENERATION

@condition.params(active=False)
def stopby_maxfitness(population):
    return False

def eval(STPG):
    def wrapps(chromosome):
        return evaluate_binary(chromosome, STPG.graph, STPG.terminals, STPG.nro_nodes, lambda k : (k-1) * 100 )

    return wrapps

def set_loggers(dst):
    pass


def test_simulation():

    population = Population(chromosomes=[random_binary(chromosome_length) for _ in range(POPULATION_SIZE)],
                            evaluation_func=eval(STPG)).evaluate().normalize()

    start = time.time()
    while condition.check(population):
        print(condition.conditions)
        print("running")
        population.select(selector_func=roullete)
        population.recombine(mate_func=crossover_2points)
        population.mutation(mutate_func=flip_nbit)
        population.evaluate()
        population.normalize()
        population.generation += 1

    runtime = time.time() - start

    print(runtime)

# test_simulation()

