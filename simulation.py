
import os
from graph import ReaderORLibrary, Graph
from ga_crossoverpartition import GeneticAlgorithm, PartitionCrossover


if __name__ == "__main__":

    dataset = os.path.join("datasets","ORLibrary","steinb13.txt")

    reader = ReaderORLibrary()

    stp = reader.parser(dataset)
    graph = Graph(vertices=stp.nro_nodes, edges=stp.graph)

    GA = GeneticAlgorithm(graph, stp.terminals)
    GA.set_crossover_operator(PartitionCrossover(graph), probability = 1)

    POPULATION_SIZE = 10
    MAX_GENERATION = 100
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


    OUTPUT_DATA = os.path.join("output_data", "simulation")
    GA.report_log(folder=OUTPUT_DATA)
