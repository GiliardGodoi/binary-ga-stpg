
from genetic.simulator import simulation
from genetic.condition import condition

from random import normalvariate

MAX_GENERATION = 100

@condition
def stopby_maxlimit(number):
    return number < MAX_GENERATION

@simulation(name="normal distribution", mean=50, std=15)
def test_simulation(max_trial, mean=0, std=5, **kwargs):

    samples = list()

    count = 0
    while condition.check(count):
        sample = normalvariate(mean, std)
        samples.append(sample)
        count += 1

    return samples
