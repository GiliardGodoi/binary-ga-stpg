# -*- coding: utf-8 -*-
"""
Created on 25 03 2020

@author: Giliard Almeida de Godoi

Funções recorrentes para a leitura, tratamento e análise dos dados.
"""

problems_class = {
        'b' : {'max' : 18},
        'c' : {'max' : 20},
        'd' : {'max' : 20},
        'e' : {'max' : 20},
        'others' : ['dv80', 'dv160', 'dv320']
    }

def generate_file_names(key = None, file_extension = "txt"):

    if isinstance(problems_class[key], list) :
        for item in problems_class[key]:
            yield item + file_extension

    elif isinstance(problems_class[key], dict) :
        counter = 1
        MAX = problems_class[key]['max']
        while counter <= MAX :
            yield f"stein{key}{counter}.{file_extension}"
            counter += 1