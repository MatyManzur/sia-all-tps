import json
import time

from global_config import config
import random
from typing import List, Callable
from classes import Warrior, Rogue, Warden, Archer, BaseClass
from crossover import crossover_population, select_cross_function
from selection import select, get_select_func
from mutation import mutate_population
from finish_criteria import get_finish_condition

AMOUNT_STATS = 5

FinishFunction = Callable[[List[BaseClass], int, float], bool]

CHAR_ARRAY = [Warrior, Rogue, Warden, Archer]


def all_chars():
    initial_config = config['initial']

    random.seed(initial_config['population_seed'])

    children_count: int = initial_config['children_count']

    pop_size = initial_config['population_size']

    select_1 = get_select_func(config['selection']['selection_1'])
    select_2 = get_select_func(config['selection']['selection_2'])
    select_ratio = config['selection']['A']

    replacement_1 = get_select_func(config['selection']['selection_3'])
    replacement_2 = get_select_func(config['selection']['selection_4'])
    replace_ratio = config['selection']['B']

    crossover = select_cross_function(config['crossover']['cross'])

    new_population: List[BaseClass] = []

    finished_function = get_finish_condition(config['finish_criteria']['finish_method'])
    start_time = time.time() * 1000

    result = {"characters": {}}
    for char in CHAR_ARRAY:
        result["characters"][char.__name__] = {}
        result["characters"][char.__name__]["all_generations"] = {}
        population: List[BaseClass] = generate_population(pop_size, char)

        finished = False
        generation = 0

        while not finished:

            population.sort(key=lambda x: x.get_fitness(), reverse=True)
            best = population[0]
            result["characters"][char.__name__]["all_generations"][f"gen_{generation}"] = {}
            result["characters"][char.__name__]["all_generations"][f"gen_{generation}"]["best"] = {
                "fitness": best.get_fitness(), "genes": best.genes}

            worst = population[-1]
            result["characters"][char.__name__]["all_generations"][f"gen_{generation}"]["worst"] = {
                "fitness": worst.get_fitness(), "genes": worst.genes}

            # Selection
            selected_pop = select(population, children_count, generation, select_1, select_2, select_ratio)
            generation += 1
            # Crossover
            random.shuffle(selected_pop)
            child_pop = crossover_population(selected_pop, crossover)

            # Mutation
            child_pop = mutate_population(child_pop,generation)

            # Replacement

            if config['selection']['replacement_method'] == 'traditional':
                new_population = select(population + child_pop, pop_size, generation,
                                        replacement_1, replacement_2, replace_ratio)
            elif config['selection']['replacement_method'] == 'young':
                if len(child_pop) > pop_size:
                    new_population = select(child_pop, pop_size, generation, replacement_1, replacement_2,
                                            replace_ratio)
                else:
                    new_population = child_pop + select(population, pop_size - len(child_pop), generation,
                                                        replacement_1, replacement_2, replace_ratio)

            population = new_population

            # Finish condition
            finished = finished_function(population, generation, time.time() * 1000 - start_time)

        end_time = time.time() * 1000

        population.sort(key=lambda x: x.get_fitness(), reverse=True)
        best = population[0]
        result["characters"][char.__name__]["all_generations"][f"gen_{generation}"] = {}
        result["characters"][char.__name__]["all_generations"][f"gen_{generation}"]["best"] = {
            "fitness": best.get_fitness(), "genes": best.genes}

        worst = population[-1]
        result["characters"][char.__name__]["all_generations"][f"gen_{generation}"]["worst"] = {
            "fitness": worst.get_fitness(), "genes": worst.genes}
        result["characters"][char.__name__]["generation_count"] = generation
        result["characters"][char.__name__]["elapsed_time"] = end_time - start_time

    with open("result_all_chars.json", "w") as outfile:
        json.dump(result, outfile)


def generate_population(n: int, create_function) -> List[BaseClass]:
    population = []
    stats = [0, 0, 0, 0, 0, 0]

    for _ in range(n):
        for j in range(AMOUNT_STATS):
            stats[j] = random.uniform(0, 150)
        stats[5] = random.uniform(1.3, 2.0)
        # ...
        population.append(create_function(*stats))
    return population


if __name__ == '__main__':
    all_chars()
