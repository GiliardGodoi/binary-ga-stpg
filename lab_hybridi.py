 #

from ga_hybridi import HybridGeneticAlgorithm
from simulator import SimulatorGA

# Basic configuration for all variations
parametrization = {
    "tx_crossover" : 0.9,
    "tx_mutation" : 0.2,
    "population_size" : 100,
    "max_generation" : 10_000,
    "change_interval" : 100
}

test = {
    "tx_crossover" : 0.9,
    "tx_mutation" : 0.2,
    "population_size" : 10,
    "max_generation" : 1000,
    "change_interval" : 50
}

simulation = SimulatorGA("teste_hybridi", test)

simulation.set_gaclass(HybridGeneticAlgorithm)
simulation.setup_dataset("steinb13.txt")
simulation.setup_ga()

simulation.run()
