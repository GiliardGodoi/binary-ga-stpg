
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
log_outputfile = os.path.join(config.log_folder, "simulation.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s : %(name)s : %(message)s")

file_handler = logging.FileHandler(log_outputfile)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

## =====================================================================
def display(*msgs, **kwargs):
    print(*msgs)

class ParameterNotFoundError(AttributeError):
    '''Parameter not defined'''

class SimulatorGA():
    '''
    SimulatorGA
    -----------

    Essa classe server para padronizar a maneira como as simulações são executadas.
    Ao final de cada algoritmo (GA) eu acabava por organziar as simulações e instânciar os parâmetros
    ao longo do arquivo.
    Isso poderia gerar inconsistência quanto a maneira de instanciar as simulações ou sobre quais parâmetros
    seriam utilizados.
    O problema é que esse código parece que gerou uma complexidade que realmente não seria necessária.
    Apesar de facilitar (ver arquivos lab_binary.py) como as simulaçoes são instânciadas.
    Eu posso até mesmo instânciar e organizar todas em um mesmo arquivo main.py e escolher qual simulaçao vou rodar
    utilizando opção por linha de comando.
    '''
    def __init__(self, name, params, *args, **kwargs):

        self.simulation_name = name
        self.params = params

        self.STPG = None
        self.GA = None

    def setup_dataset(self, filename, **kwargs):
        filename = os.path.join(config.datasets_folder, filename)
        reader = ReaderORLibrary()
        self.STPG = reader.parser(filename)

        logger.debug(f"setting dataset {filename} | nodes {self.STPG.nro_nodes} | edges {self.STPG.nro_edges} | terminals {self.STPG.nro_terminals}")

    def setup_ga(self, *args, **kwargs):

        GAClass = self.GAclass
        ## inicialize properly GA class
        self.GA = GAClass(self.STPG, self.params)

        logger.debug(f"setting GA {GAClass.__name__} | STPG {self.STPG.name} | params {str(self.params)}")
        self.setup_datalogger(**kwargs)

    def setup_datalogger(self, **kwargs):
        folder = os.path.join(config.output_folder, self.simulation_name, self.STPG.name)
        trial = str(kwargs.get("trial", ''))
        self.GA.datacolector = DataLogger(prefix=f"trial_{trial}", outputfolder=folder)

        logger.debug(f"setting {self.GA.datacolector.__class__.__name__} | trial {trial} | outputfolder {folder}")

    def set_gaclass(self, GAclass):
        self.GAclass = GAclass

    def set_stop_condition(self, function):
        '''
        Seria interessante termos formas de definir diversos critérios de parada.
        Por enquanto a maior dificuldade é lidar com a variabilidade. Como as coisas podem variar.

        A própria funcao que expressa o critério de parada precisa ser parametrizadas de alguma forma:
        como por exemplo, qual o máximo de gerações permitidas antes que ela retorne False para *time out*.
        E se a função indicar uma parada por ter atingido o tempo máximo de execução (tempo medido em segundos).

        Quais os critérios de parada possíveis:
            - atingiu o ótimo global relatado na literatura
                informar qual é o máximo global conhecido
            - atingiu o número máximo de gerações
                informar qual é esse número máximo de iterações
            - por estagnação: por exemplo o algoritmo não conseguiu aprimorar a solução dentro de um intervalo de tempo
                como informar essse intervalo de tempo

        Quais outros critérios poderiam ser definidos?

        Como poderiamos definir esses critérios ou como podemos setados para simulation?
        poderiamos ter uma função que recebe o dicionário de parâmetros (definidos em lab_*.py) e retornar uma função
        que expressa o critério de parada.

            parameters = {
                'max_generation' : 10_000
                ...
            }

            def init_stop_condition(params):

                max_generation = params["max_generation"]

                def still_running(iteration=0, **kwargs):
                    O retorno deve ser padronizado:
                        (boolean[True | False], string )
                    return iteration < max_generation, "max_generation_reached"


                return still_running

            simulation.set_stop_condition(init_stop_condition(parameters))
        '''
        raise NotImplementedError("Método não implementado ainda")

    def get_stop_condition(self):

        max_generation = self.params["max_generation"]

        def default_stop_condition(iteration=0, **kwargs):
            return iteration < max_generation, "always"

        return default_stop_condition

    def check_parameters(self, GAinstance, add_names = []):
        '''
        Essa função não esta sendo utilizada, mas ela foi projetada para garantir,
        no momento da inicialização da simulação, que todos os parâmetros necessários para rodar o GA
        estão lá, ou foram definidos.
        Um bom local que isso poderia ficar é na função de setupGA.
        '''
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
            if not hasattr(GAinstance, attr):
                raise ParameterNotFoundError(f"Attribute Not Found {attr}")

    def run(self, *args, **kwargs):

        optimum = kwargs.get("global_optimum", 0)
        iteration = 0
        stop_condition = self.get_stop_condition()

        logger.debug("STARTING GA EXECUTION")
        GA = self.GA
        GA.generate_population()

        running, why_stopped = stop_condition(iteration=iteration, global_optimum=optimum)

        start_at = time.time()
        while running:
            if iteration % 100 == 0: display(f"iterarion {iteration}")
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
        logger.debug(f"Output: {str(output)}")
        self.report_it(**output)

    def run_multiply_trials(self, DATASETS, NUMBER_OF_TRIALS, *args, **kwargs):
        '''Run multiplus trials of the experiment'''
        for filename, global_optimum in DATASETS:
            logger.info(f"Executing    {filename}")
            self.setup_dataset(filename)
            for nro in range(1, NUMBER_OF_TRIALS + 1):
                self.setup_ga(trial=nro)
                logger.info(f"Executing     {filename}    Trial    {nro}")
                self.run(global_optimum=global_optimum)

    def report_it(self, *args, **kwargs):

        self.GA.datacolector.log("simulation",
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
        self.GA.datacolector.report()

    def report_solution_found(self):
        self.GA.datacolector.register("solution", "json")

        if type(self.GA.best_chromosome) is BinaryChromosome:
            tree_chromo = convert_binary2treegraph(self.GA.best_chromosome, self.GRAPH, self.STPG.terminals, self.STPG.nro_nodes)

            self.GA.datacolector.log("solution", tree_chromo.genes.edges)

        elif type(self.GA.best_chromosome) is TreeBasedChromosome:
            self.GA.datacolector.log("solution", self.GA.best_chromosome.genes.edges)
