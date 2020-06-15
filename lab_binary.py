from genetic.binary import random_binary
from genetic.binary.crossover import crossover_2points, crossover_uniform
from genetic.binary.mutate import flip_nbit
from genetic.binary.selection import roullete
from genetic.condition import condition
from simulation import simulation

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

if __name__ == "__main__":

    DATASETS = [
        ("steinb15.txt", 318), # 14
        ("steinb16.txt", 127), # 15
        ("steinb17.txt", 131), # 16
        ("steinb18.txt", 218), # 17
    ]

    for dataset, value in DATASETS:
        simulation(population_size= 100,
                n_iterations= 10_000,
                n_trials = 30,
                improvement_interval = 500,
                stpg_filename = dataset,
                best_known_solution = value,
                evol_func = pipe_crossoveruniform,
                simulation_name = "20200615_cxuniform_interval500"
            )

    for dataset, value in DATASETS:
        simulation(population_size= 100,
                n_iterations= 10_000,
                n_trials = 30,
                improvement_interval = 1000,
                stpg_filename = dataset,
                best_known_solution = value,
                evol_func = pipe_crossoveruniform,
                simulation_name = "20200615_cxuniform_interval1000"
            )


    for dataset, value in DATASETS:
        simulation(population_size= 100,
                n_iterations= 10_000,
                n_trials = 30,
                improvement_interval = 500,
                stpg_filename = dataset,
                best_known_solution = value,
                evol_func = pipe_crossover2points,
                simulation_name = "20200615_cx2points_interval500"
            )

    for dataset, value in DATASETS:
        simulation(population_size= 100,
                n_iterations= 10_000,
                n_trials = 30,
                improvement_interval = 1000,
                stpg_filename = dataset,
                best_known_solution = value,
                evol_func = pipe_crossoveruniform,
                simulation_name = "20200615_cx2points_interval1000"
            )

