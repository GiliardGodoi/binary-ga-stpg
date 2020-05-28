
from ga_simplestpartition import PXSimpliestGeneticAlgorithm
from simulator import SimulatorGA

# Basic configuration for all variations
parametrization = {
    "tx_crossover" : 0.9,
    "tx_mutation" : 0.2,
    "population_size" : 100,
    "max_generation" : 10_000,
}

test = {
    "tx_crossover" : 0.9,
    "tx_mutation" : 0.2,
    "population_size" : 10,
    "max_generation" : 1000,
}

simulation = SimulatorGA("teste_hybridi", test)

simulation.set_gaclass(PXSimpliestGeneticAlgorithm)
simulation.setup_dataset("steinb13.txt")
simulation.setup_ga()

simulation.run()