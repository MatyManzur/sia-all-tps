import math
import random

import numpy
import numpy as np
from typing import Set, List, Callable
from classes import BaseClass
from global_config import config, get_config

selection_options = config['selection']

T_C = selection_options['t_c']
T_0 = selection_options['t_0']
T_K = selection_options['t_k']
TOURNAMENT_M = selection_options['tournament_m']
TOURNAMENT_THRESHOLD = selection_options['tournament_threshold']

SelectionFunction = Callable[[List[BaseClass], int, int], List[BaseClass]]


def change_selection_config():
    global selection_options, T_C, T_0, T_K, TOURNAMENT_M, TOURNAMENT_THRESHOLD
    selection_options = get_config()['selection']
    T_C = selection_options['t_c']
    T_0 = selection_options['t_0']
    T_K = selection_options['t_k']
    TOURNAMENT_M = selection_options['tournament_m']
    TOURNAMENT_THRESHOLD = selection_options['tournament_threshold']


def elite(characters: List[BaseClass], n, t) -> List[BaseClass]:
    best_indexes = np.argsort(characters)[-n:]
    return [characters[i] for i in best_indexes]


def roulette(chars: List[BaseClass], n, t) -> List[BaseClass]:
    total_fitness = 0
    for c in chars:
        total_fitness += c.get_fitness()

    accumulated_fitness = []
    current_accumulated = 0
    for c in chars:
        current_accumulated += c.get_fitness() / total_fitness
        accumulated_fitness.append(current_accumulated)

    accumulated_fitness[len(accumulated_fitness) - 1] = 1

    selected_n = []
    for i in range(n):
        r = random.uniform(0, 1)
        last = 0
        for j, acc in enumerate(accumulated_fitness):
            if last < r <= acc:
                selected_n.append(chars[j])
                break
            last = acc
    return selected_n


def universal(chars: List[BaseClass], n, t) -> List[BaseClass]:
    total_fitness = 0
    for c in chars:
        total_fitness += c.get_fitness()

    accum_aux = 0
    accumulated = []
    for c in chars:
        accum_aux += c.get_fitness() / total_fitness
        accumulated.append(accum_aux)

    accumulated[len(accumulated) - 1] = 1

    ret_list = []
    r = random.uniform(0, 1)
    for i in range(n):
        r_i = (r + i) / n
        last = 0
        for j, acc in enumerate(accumulated):
            if last < r_i <= acc:
                ret_list.append(chars[j])
                break
            last = acc

    return ret_list


def __temperature(k, t):
    return T_C + (T_0 - T_C) * (math.e ** (-T_K * t))


# t is the generation number
def boltzmann(chars: List[BaseClass], n, t) -> List[BaseClass]:
    exp_vals = numpy.zeros(len(chars), dtype=np.float_)
    for i, char in enumerate(chars):
        exp_vals[i] = math.e ** (char.get_fitness() / __temperature(n, t))

    exp_vals /= numpy.average(exp_vals)

    total_fitness = numpy.sum(exp_vals)
    accumulated_fitness = []
    current_accumulated = 0
    for i in range(n):
        current_accumulated += exp_vals[i] / total_fitness
        accumulated_fitness.append(current_accumulated)

    accumulated_fitness[len(accumulated_fitness) - 1] = 1

    selected_n = []
    for i in range(n):
        r = random.uniform(0, 1)
        last = 0
        for j, acc in enumerate(accumulated_fitness):
            if last < r <= acc:
                selected_n.append(chars[j])
                break
            last = acc
    return selected_n


def ranking(chars: List[BaseClass], n, t) -> List[BaseClass]:
    # Ruleta pero primero armo un ranking
    chars.sort(reverse=True)
    fitness_sim = []
    amount_char = len(chars)

    sum_value = 0
    for index, c in enumerate(chars):
        f = (amount_char - (index + 1)) / amount_char
        fitness_sim.append(f)
        sum_value += f

    # hace falta esto? No podemos hacer de 0 a sum_value y listo?
    fitness_sim_accum = []
    accumulated = 0
    for val in fitness_sim:
        fitness_sim_accum.append((accumulated + val) / sum_value)
        accumulated += val

    fitness_sim_accum[len(fitness_sim_accum) - 1] = 1

    new_population = []
    for _ in range(n):
        r = random.uniform(0, 1)
        last = 0
        for index, acc in enumerate(fitness_sim):
            if last < r <= acc:
                new_population.append(chars[index])
                break

    return new_population


def deterministic_tournament(chars: List[BaseClass], n, t) -> List[BaseClass]:
    selected_n = []
    for i in range(n):
        match_chars = random.sample(chars, TOURNAMENT_M)
        selected_n.append(max(match_chars))
    return selected_n


def probabilistic_tournament(chars: List[BaseClass], n, t) -> List[BaseClass]:
    selected_n = []
    for i in range(n):
        bicho_one = chars[random.randint(0, len(chars) - 1)]
        bicho_two = chars[random.randint(0, len(chars) - 1)]
        r = random.uniform(0, 1)
        if r < TOURNAMENT_THRESHOLD:
            selected_n.append(max(bicho_one, bicho_two))
        else:
            selected_n.append(min(bicho_one, bicho_two))
    return selected_n


def get_select_func(string: str) -> SelectionFunction:
    if string == 'elite':
        return elite
    elif string == 'roulette':
        return roulette
    elif string == 'universal':
        return universal
    elif string == 'boltzmann':
        return boltzmann
    elif string == 'ranking':
        return ranking
    elif string == 'deterministic_tournament':
        return deterministic_tournament
    elif string == 'probabilistic_tournament':
        return probabilistic_tournament
    else:
        raise Exception('Invalid Selection Function Name!')


def select(population: List[BaseClass], n, generation, select_fun_1: SelectionFunction,
           select_fun_2: SelectionFunction, sel_fun_1_part: float):
    selected_1 = select_fun_1(population, int(n * sel_fun_1_part), generation)
    selected_2 = select_fun_2(population, n - len(selected_1), generation)
    return selected_1 + selected_2
