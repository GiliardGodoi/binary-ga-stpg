import csv
import json
import os
from collections import defaultdict


class BaseLogger():
    '''Default logger. It doesn't do anything. It's just for prevent errors.
    Based in idea from Design Partterns Null Object.
    '''

    def __init__(self, prefix='', outputfolder='outputdata'):
        pass

    def add_header(self, key, *args):
        pass

    def log(self, key, *args):
        pass

    def report(self):
        pass

class SimulationLogger(BaseLogger):
    '''Simple class to collect and store data from the simulations.

    Notes:
    1. Guarda os dados solicitados em memória e
    depois persiste em um arquivo csv no disco.
    Não faz o gerenciamento da quantidade de registros em memória.

    2. Não faz verificação da quantidade de registro passados.

    A intenção é reutilizar esse código nos demais módulos de simulações.
    '''

    ## define um objeto que será compartilhado por todas as instâncias
    __default_state = dict()

    def __init__(self, prefix='', outputfolder='default'):
        if outputfolder and (not os.path.exists(outputfolder)):
            os.mkdir(outputfolder)

        self.__dict__ = self.__default_state

        if 'storage' not in self.__dict__:
            self.storage = defaultdict(list)

        self.mainfolder = outputfolder
        self.prefix = prefix

    def add_header(self, key, *args):
        if not key:
            raise TypeError("Error at <key> parameter")
        self.storage[key].append(args)

    def log(self, key, *args):
        if not key:
            raise TypeError("Error at <key> parameter")
        if key not in self.storage:
            raise TypeError(f"Não foi registrado cabeçalho para: {key}")

        self.storage[key].append(args)

    def report(self):
        prefix = f'{self.prefix}_' if self.prefix else ''
        mainfolder = self.mainfolder if self.mainfolder else '.'

        for key, data in self.storage.items():
            filename = os.path.join(mainfolder, f'{prefix}{key}')

            if isinstance(data, (list, tuple)):
                self.__write_csv__(filename, data)
            elif isinstance(data, dict):
                self.__write_json__(filename, data)

    def __write_csv__(self, filename, data, mode='w'):

        if mode not in ['w', 'a', 'x']:
            raise TypeError("Mode must be 'w', 'a' or 'x' ")

        filename = self.__enforce_extension__(filename, enforce_extension='.csv')

        try:
            with open(filename, mode, newline='') as file :
                writer = csv.writer(file)
                writer.writerows(data)
        except Exception as msg:
            print(msg)
            return False

        return True

    def __write_json__(self, filename, data, mode='w'):

        if mode not in ['w', 'a', 'x']:
            raise TypeError("Mode must be w or w+")

        if not isinstance(data, dict):
            print("")

        filename = self.__enforce_extension__(filename, enforce_extension='.json')

        try:
            with open(filename, mode) as file :
                json.dump(data, fp=file, indent=4)
        except Exception as msg:
            print(msg)
            return False

        return True

    def __enforce_extension__(self, filename, enforce_extension='.txt'):
        #enforces the extension
        if not filename.endswith(enforce_extension):
            if '.' in filename:
                extension = filename[filename.rfind('.'):]
                filename = filename.replace(extension, enforce_extension)
            else:
                filename += enforce_extension

        return filename


if __name__ == "__main__":
    from tqdm import tqdm
    import random


    logger = SimulationLogger(outputfolder=None)
    logger.add_header("teste",'i', 'int', 'measure', 'category', 'temp')

    for i in tqdm(range(100000)):
        logger.log("teste", i, random.randint(50,1000), random.random() * 100, random.choice(['A', 'B', 'C']), random.random() * 100)


    logger.report()
