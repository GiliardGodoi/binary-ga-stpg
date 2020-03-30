from collections import defaultdict
import csv
import os

class Logger():
    '''Default logger. It doesn't do anything. It's just for prevent errors.
    Based in idea from Design Partener Null Object.
    '''

    def __init__(self, prefix='', outputfolder='outputdata'):
        pass

    def add_header(self, key, *args):
        pass

    def log(self, key, *args):
        pass

    def report(self):
        pass

class SimulationLogger(Logger):
    '''Simple class to collect and store data from the simulations.

    Notes:
    1. Guarda os dados solicitados em memória e
    depois persiste em um arquivo csv no disco.
    Não faz o gerenciamento da quantidade de registros em memória.

    2. Não faz verificação da quantidade de registro passados.

    A intenção é reutilizar esse código nos demais módulos de simulações.
    '''
    def __init__(self, prefix='', outputfolder='default'):
        self.prefix = prefix
        self.mainfolder = outputfolder
        if outputfolder and (not os.path.exists(outputfolder)):
            os.mkdir(outputfolder)

        self.storage = defaultdict(list)

    def add_header(self, key, *args):
        self.log(key, *args)

    def log(self, key, *args):
        if not key:
            raise AttributeError("Error at <key> parameter")
        self.storage[key].append(args)

    def report(self):
        prefix = f'{self.prefix}_' if self.prefix else ''
        mainfolder = self.mainfolder if self.mainfolder else '.'

        for key, data in self.storage.items():
            filename = os.path.join(mainfolder, f'{prefix}{key}.csv')
            with open(filename,"w", newline="") as file :
                writer = csv.writer(file)
                writer.writerows(data)


if __name__ == "__main__":
    from tqdm import tqdm
    import random


    logger = SimulationLogger(outputfolder=None)
    logger.add_header("teste",'i', 'int', 'measure', 'category', 'temp')

    for i in tqdm(range(100000)):
        logger.log("teste", i, random.randint(50,1000), random.random() * 100, random.choice(['A', 'B', 'C']), random.random() * 100)


    logger.report()
