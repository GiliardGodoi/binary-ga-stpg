
from graph import Reader

from os import path

diretorio_dados = "datasets"
arquivo_dados = "b01.stp"

arquivo = path.join(diretorio_dados, arquivo_dados)

reader = Reader()

stp = reader.parser(arquivo)

import pprint as pp 

pp.pprint(stp.graph)