
from random import choice, random, sample
from bisect import bisect
from itertools import cycle

def crossover_2points(parent_a, parent_b):
    length_a, length_b = len(parent_a), len(parent_b)
    assert length_a == length_b, "chromosomes doesn't have the same length"
    points = sample(range(0, length_a), k=2)
    points.sort()
    p1, p2 = points

    return (parent_a[:p1] + parent_b[p1:p2] + parent_a[p2:])

def crossover_1point(parent_a, parent_b):
    length_a, length_b = len(parent_a), len(parent_b)
    assert length_a == length_b, "chromosomes doesn't have the same length"
    index = choice(range(0, length_a))
    return (parent_a[:index] + parent_b[index:])

def crossover_Npoints(parent_a, parent_b, n=2):
    length_a, length_b = len(parent_a), len(parent_b)
    assert length_a == length_b, "chromosomes doesn't have the same length"
    assert n < (length_a - 1) , f"It is allowed only {(length_a - 1)} cuts. It was given {n}"

    choose_parents = cycle("AB")
    parents = [ next(choose_parents) for _ in range(n+1) ]
    breakpoints = sample(range(0,length_a+1), k=n)
    breakpoints.sort()

    def choose(idx):
        return parents[bisect(breakpoints, idx)]

    newchromosome = [ genes[0] if choose(idx) == "A" else genes[1] for idx, genes in enumerate(zip(parent_a, parent_b))]

    if isinstance(parent_a, str):
        newchromosome = ''.join(newchromosome)

    return newchromosome

def crossover_nbreakpoints(parent_a, parent_b, n=2):

    raise RuntimeWarning("It doesnt works for n > 2")
    length_a, length_b = len(parent_a), len(parent_b)
    assert length_a == length_b, "chromosomes doesn't have the same length"
    assert n <= (length_a - 1) , f"It is allowed only {(length_a - 1)} cuts. It was given {n}"

    def chunck(parent,start, end):
        return parent[start:end]

    points = [0] + sample(range(0, length_a+1), k=n) + [length_a]
    flag = True
    newchromosome = list()
    for start, end in zip(points, points[1:]):
        if flag:
            newchromosome.append(chunck(parent_a, start, end))
        else:
            newchromosome.append(chunck(parent_b, start, end))
        flag = not flag

    return ''.join(newchromosome)


def crossover_uniform(chromosome_a, chromosome_b, pbcrossover=0.5, **kwargs):

    assert len(chromosome_a) == len(chromosome_b), "chromosome must have the same length"

    # list comprehension
    chromosome = [ a if random() < pbcrossover else b
                        for a, b in zip(chromosome_a, chromosome_b) ]

    # desta forma se o chromosomo receber uma lista de números inteiros ele também vai funcionar
    if isinstance(chromosome_a, str) or isinstance(chromosome_b, str):
        chromosome = ''.join(chromosome)

    return chromosome
