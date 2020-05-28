import os

'''
Parametrização geral das simulações

Existem diversas formar de criar arquivos de parametrização.
A vantagem de usar scripts é que podemos programar nos arquivos.

@author: Giliard

Referencias:
------------
https://docs.python.org/3/library/configparser.html
https://martin-thoma.com/configuration-files-in-python/

'''

## =======================================================
## Simulator.py configurations

datasets_folder = os.path.join("datasets", "ORlibrary")
output_folder = "outputdata"
log_folder = "log"

report_simulation = True
report_bestsolution = True