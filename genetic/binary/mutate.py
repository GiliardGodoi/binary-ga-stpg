
from random import randrange, random

def flip_onebit(chromosome, defaults=['0', '1']):
    flip = lambda x : defaults[1] if x == defaults[0] else defaults[0]

    index = randrange(len(chromosome))

    return chromosome[:index] + flip(chromosome[:index]) + chromosome[index+1:]

def flip_nbit(chromosome, flip_probability=0.2, defaults=['0', '1']):

    flip = lambda x : defaults[1] if x == defaults[0] else defaults[0]

    newchromosome = [ flip(gene) if random() < flip_probability else gene for gene in chromosome]

    if isinstance(chromosome, str):
        newchromosome = ''.join(newchromosome)

    return newchromosome