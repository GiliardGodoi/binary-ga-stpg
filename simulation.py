
from os import path
from graph import ReaderORLibrary, Graph
from crossover_partition import GeneticAlgorithm, PartitionCrossover


if __name__ == "__main__":

    arquivo = path.join("datasets","ORLibrary","steinb18.txt")

    reader = ReaderORLibrary()

    stp = reader.parser(arquivo)
    graph = Graph(vertices=stp.nro_nodes, edges=stp.graph)

    GA = GeneticAlgorithm(graph, stp.terminals)
    GA.set_crossover_operator(PartitionCrossover(graph), probability = 1)

    POPULATION_SIZE = 10
    MAX_GENERATION = 1000
    iteration = 0

    GA.initial_population(POPULATION_SIZE)
    GA.sort_population()

    for c in GA.population:
        print(c)

    while iteration < MAX_GENERATION:
        print("Iteration: ", (iteration + 1), end="\r")
        GA.evaluate()
        # GA.normalized_fitness()
        GA.selection()
        GA.recombine()
        iteration += 1

    print("\n\n=============================\n\n")
    print(GA.best_chromossome)

