
from random import sample, choice

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
    points = sample(range(0,length_a), k=n)

    raise NotImplementedError