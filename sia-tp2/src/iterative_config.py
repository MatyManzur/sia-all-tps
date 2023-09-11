import json
import time

from global_config import get_config, change_config_file
import random
from typing import List, Callable
from classes import Warrior, Rogue, Warden, Archer, BaseClass
from crossover import crossover_population, select_cross_function
from selection import select, get_select_func
from mutation import mutate_population
from finish_criteria import get_finish_condition

from crossover import change_cross_config
from mutation import change_mutation_config
from selection import change_selection_config
from finish_criteria import change_finish_config

AMOUNT_STATS = 5

FinishFunction = Callable[[List[BaseClass], int, float], bool]
# Comentarlos si no se usan, para asi no copiamos devuelta
# CONFIG_FILES = ["configs/config_complete.json", "configs/config_gen.json", "configs/config_limited.json",
#                 "configs/config_uniform.json", "configs/config_decreasing.json", "configs/config_increasing.json",
#                 "configs/config_sinusoidal.json"]

# RESULT_NAMES = ["results/complete.json", "results/gen.json", "results/limited.json", "results/uniform.json",
#                 "results/decreasing.json", "results/increasing.json", "results/sinusoidal.json"]
CONFIG_FILES = ["configs/config_anular.json","configs/config_one.json","configs/config_two.json",
                "configs/config_uniform_cross.json"]
RESULT_NAMES = ["results/anular.json","results/one.json","results/two.json","results/uniform_cross.json"]

LOOP_COUNT = 15

CHAR_ARRAY = [Warrior, Rogue, Warden, Archer]


def iterative_config():
    for i, file in enumerate(CONFIG_FILES):

        config = change_config_file(file)
        change_cross_config()
        change_mutation_config()
        change_selection_config()
        change_finish_config()

        initial_config = config['initial']
        random.seed(initial_config['population_seed'])

        pop_size = initial_config['population_size']

        children_count: int = initial_config['children_count']

        select_1 = get_select_func(config['selection']['selection_1'])
        select_2 = get_select_func(config['selection']['selection_2'])
        select_ratio = config['selection']['A']

        replacement_1 = get_select_func(config['selection']['selection_3'])
        replacement_2 = get_select_func(config['selection']['selection_4'])
        replace_ratio = config['selection']['B']

        crossover = select_cross_function(config['crossover']['cross'])

        new_population: List[BaseClass] = []

        finished_function = get_finish_condition(config['finish_criteria']['finish_method'])

        file_data = {"all_iterations": []}
        for char in CHAR_ARRAY:
            population = generate_population(pop_size, char)
            for j in range(LOOP_COUNT):
                start_time = time.time() * 1000
                finished = False
                generation = 0

                while not finished:

                    # Selection
                    selected_pop = select(population, children_count, generation, select_1, select_2, select_ratio)
                    generation += 1
                    # Crossover
                    random.shuffle(selected_pop)
                    child_pop = crossover_population(selected_pop, crossover)

                    # Mutation
                    child_pop = mutate_population(child_pop, generation)

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
                best = max(population, key=lambda x: x.get_fitness())
                print(f"Iteration {j + 1} of {file} with {char.__name__}")
                file_data["all_iterations"].append(
                    {"fitness": best.get_fitness(), "genes": best.genes, "time": time.time() * 1000 - start_time,
                     "generations": generation, "class": best.__class__.__name__})

        output_file = RESULT_NAMES[i]
        print(f"Writing to {output_file}")
        with open(output_file, "w") as outfile:
            json.dump(file_data, outfile)


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
    iterative_config()
