import math
import random
from typing import Callable, Tuple, List

from classes import Chromosome, BaseClass
from global_config import config

# traerlo del config
PARAMETER_UNIFORM_PROBABILITY = config['crossover']['uniform_probability']

CrossFunction = Callable[[Chromosome, Chromosome], Tuple[Chromosome, Chromosome]]

crossover_options = config['crossover']


def select_cross_function(name: str) -> CrossFunction:
    if name == 'one_point':
        return one_point_cross
    if name == 'two_point':
        return two_point_cross
    if name == 'anular_cross':
        return anular_cross
    if name == 'uniform_cross':
        return uniform_cross
    raise Exception('Invalid Crossover Function Name!')


def crossover_population(population: List[BaseClass],cross_function: CrossFunction) -> List[BaseClass]:
    n = len(population)
    crossed_population = []
    character_class = population[0].__class__
    for i in range(0, n, 2):
        if i == n - 1:
            new_chromies = cross_function(population[i].get_cromies(), population[0].get_cromies()) 
        else:
            new_chromies = cross_function(population[i].get_cromies(), population[i+1].get_cromies())
        population.append(character_class(chromosome=new_chromies[0]))
        population.append(character_class(chromosome=new_chromies[1]))
    return crossed_population


def one_point_cross(tuple1: Chromosome, tuple2: Chromosome) -> (Chromosome, Chromosome):
    switch = int(round(random.uniform(0, len(tuple1) - 1), 0))
    return _generic_cross(tuple1, tuple2, switch, len(tuple1))


def two_point_cross(tuple1: Chromosome, tuple2: Chromosome) -> (Chromosome, Chromosome):
    switch1 = int(round(random.uniform(0, len(tuple1) - 1), 0))
    switch2 = int(round(random.uniform(0, len(tuple1) - 1), 0))
    if switch1 > switch2:
        temp = switch2
        switch2 = switch1
        switch1 = temp
    return _generic_cross(tuple1, tuple2, switch1, switch2)


def anular_cross(tuple1: Chromosome, tuple2: Chromosome) -> (Chromosome, Chromosome):
    switch = int(round(random.uniform(0, len(tuple1) - 1), 0))
    size = int(round(random.uniform(0, math.ceil(len(tuple1) / 2)), 0))
    switch2 = (switch + size) % len(tuple1)
    if switch <= switch2:
        return _generic_cross(tuple1, tuple2, switch, switch2)
    else:
        return _generic_cross(tuple2, tuple1, switch2, switch)


def uniform_cross(tuple1: Chromosome, tuple2: Chromosome) -> (Chromosome, Chromosome):
    c1 = list(tuple1)
    c2 = list(tuple2)
    for i in range(len(c1)):
        rand = random.random()
        if rand < PARAMETER_UNIFORM_PROBABILITY:
            temp = c1[i]
            c1[i] = c2[i]
            c2[i] = temp
    return tuple(c1), tuple(c2)


# Hace cross lo que está entre switch1 y switch 2 EXCLUSIVE switch2
def _generic_cross(tuple1: Chromosome, tuple2: Chromosome, switch1: int, switch2: int):
    return (tuple(i for i in tuple1[:switch1] + tuple2[switch1:switch2] + tuple1[switch2:]),
            tuple(i for i in tuple2[:switch1] + tuple1[switch1:switch2] + tuple2[switch2:]))
