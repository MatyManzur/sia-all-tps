from global_config import config
import random
from typing import List, Callable
from classes import Warrior,Rogue,Warden,Archer,BaseClass
from crossover import crossover_population
from selection import select
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
    population = generate_population(initial_config['population_size'], initial_config['population_seed'],
                                     initial_config['class'])
    children_count: int = initial_config['children_count']


    finished = False
    generation = 0

    while not finished:
        # Selection
        selected_pop = select(population, children_count, generation)
        generation += 1

        # Crossover
        child_pop = crossover_population(selected_pop)

        # Mutation
        child_pop = mutate_population(child_pop)

        # Replacement
        

        # TODO: qué onda lo de sesgo joven, tradicional y eso
        # Finish condition


def generate_population(n: int, seed: int, character_class: str) -> List[BaseClass]:
    random.seed(seed)
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
        population.append(create_function(*stats)) # POINTERS IN PYTHON BABY, LET'S GOOOOOOOO
    return population


if __name__ == '__main__':
    main()
