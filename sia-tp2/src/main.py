import time

from global_config import config
import random
from typing import List, Callable
from classes import Warrior, Rogue, Warden, Archer, BaseClass
from crossover import crossover_population, select_cross_function
from selection import select, get_select_func
from mutation import mutate_population

AMOUNT_STATS = 5

"""
Poblacion
Generacion
Tiempo de ejecucion
Devuelve true si termino
"""
FinishFunction = Callable[[List[BaseClass], int, float], bool]


def main():
    initial_config = config['initial']
    random.seed(initial_config['seed'])

    population = generate_population(initial_config['population_size'], initial_config['class'])
    children_count: int = initial_config['children_count']


    finished = False
    generation = 0

    pop_size = initial_config['population_size']

    select_1 = get_select_func(config['selection']['selection_1'])
    select_2 = get_select_func(config['selection']['selection_2'])
    select_ratio = config['selection']['A']

    replacement_1 = get_select_func(config['selection']['selection_3'])
    replacement_2 = get_select_func(config['selection']['selection_4'])
    replace_ratio = config['selection']['B']

    crossover = select_cross_function(config['crossover']['cross'])

    new_population = []

    finished_function = get_finish_condition(config['finish_criteria']['finish_method'])
    start_time = time.time()

    while not finished:
        # Selection
        selected_pop = select(population, children_count, generation, select_1, select_2, select_ratio)
        generation += 1

        # Crossover
        child_pop = crossover_population(selected_pop)

        # Mutation
        child_pop = mutate_population(child_pop)

        # Replacement

        if config['selection']['replacement_method'] == 'traditional':
            new_population = select(population + child_pop, pop_size, generation,
                                    replacement_1, replacement_2, replace_ratio)
        elif config['selection']['replacement_method'] == 'young':
            if len(child_pop) > pop_size:
                new_population = select(child_pop, pop_size, generation, replacement_1, replacement_2, replace_ratio)
            else:
                new_population = child_pop + select(population, pop_size - len(child_pop), generation,
                                                    replacement_1, replacement_2, replace_ratio)

        print(new_population)

        # Finish condition


def generate_population(n: int, character_class: str) -> List[BaseClass]:
    population = []

    if character_class == 'warrior':
        create_function = Warrior.__init__
    elif character_class == 'rogue':
        create_function = Rogue.__init__
    elif character_class == 'warden':
        create_function = Warden.__init__
    elif character_class == 'archer':
        create_function = Archer.__init__
    else:
        raise Exception('Invalid character class')

    stats = [0, 0, 0, 0, 0, 0]

    for _ in range(n):
        for j in range(AMOUNT_STATS):
            stats[j] = random.uniform(0, 150)
        stats[5] = random.uniform(1.3, 2.0)
        # ...
        population.append(create_function(*stats))  # POINTERS IN PYTHON BABY, LET'S GOOOOOOOO
    return population


if __name__ == '__main__':
    main()
