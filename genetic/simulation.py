import time
import os

from graph.reader import ReaderORLibrary

class Simulation:

    def __init__(self):
        pass

    def set_strategy(self):
        pass

    def set_paramns(self):
        pass

    def build(self):
        pass

    def run_trials(self, dataset, nro_trial):

        for i in range(1, nro_trial+1):
            self.run(dataset, trial=i)

    def run(self, dataset, trial=0):
        '''Run a simulation trial'''

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

