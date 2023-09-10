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


def main():
    initial_config = config['initial']
    random.seed(initial_config['population_seed'])

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

    new_population: List[BaseClass] = []

    finished_function = get_finish_condition(config['finish_criteria']['finish_method'])
    start_time = time.time() * 1000

    result = {"all_generations": {}}

    while not finished:

        population.sort(key=lambda x: x.get_fitness(), reverse=True)
        result["all_generations"][f"gen_{generation}"] = {
            "population": list(map(lambda p: {"fitness": p.get_fitness(), "genes": p.genes}, population))
        }

        # Selection
        selected_pop = select(population, children_count, generation, select_1, select_2, select_ratio)
        generation += 1
        print(generation)
        print(max(selected_pop, key=lambda x: x.get_fitness()).get_fitness())

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
                new_population = select(child_pop, pop_size, generation, replacement_1, replacement_2, replace_ratio)
            else:
                new_population = child_pop + select(population, pop_size - len(child_pop), generation,
                                                    replacement_1, replacement_2, replace_ratio)

        population = new_population

        # Finish condition
        finished = finished_function(population, generation, time.time() * 1000 - start_time)

    end_time = time.time() * 1000

    population.sort(key=lambda x: x.get_fitness(), reverse=True)
    result["all_generations"][f"gen_{generation}"] = {
        "population": list(map(lambda p: {"fitness": p.get_fitness(), "genes": p.genes}, population))
    }
    result["generation_count"] = generation
    result["elapsed_time"] = end_time - start_time

    print(f"Generation_count: {generation} \nElapsed Time: {result['elapsed_time']}")
    print(f"Top 10 of last generation:")
    for i in range(10):
        print(f"{i}: {population[i].get_fitness()} - {population[i]}")

    with open("result.json", "w") as outfile:
        json.dump(result, outfile)


def generate_population(n: int, character_class: str) -> List[BaseClass]:
    population = []
    if character_class == 'warrior':
        create_function = Warrior
    elif character_class == 'rogue':
        create_function = Rogue
    elif character_class == 'warden':
        create_function = Warden
    elif character_class == 'archer':
        create_function = Archer
    else:
        raise Exception('Invalid character class')

    stats = [0, 0, 0, 0, 0, 0]

    for _ in range(n):
        for j in range(AMOUNT_STATS):
            stats[j] = random.uniform(0, 150)
        stats[5] = random.uniform(1.3, 2.0)
        # ...
        population.append(create_function(*stats))
    return population


if __name__ == '__main__':
    main()

"""
    Ideas para gráficos
        Progreso de performance x generacion
            Por clase
            Por metodo de seleccion
            Por metodo de crossover
            Por metodo de mutacion
            Por population size
            Por promiscuidad
        Mejores de cada clase
            Por cantidad de generaciones
            Tiempo
            Estructura
            Cierto Fitness
        Énfasis en la idea de Exploración vs Explotación
        Resultados por metodos de corte
        Resultados por mutacion
            Tiempo de ejecucion
            Fitness
            Variabilidad genetica       

        Habría que ver qué métodos de selección/reemplazo/crossover/mutación son mejores para exploración y cuáles son mejores
        para explotación. Así no hacemos todas las combinaciones, sino que separamos en combinaciones más para exploración y más para
        explotación. 
        Algunas combinaciones interesantes: 
        1- Universal y Prob_Tournament para sel 1 y sel 2. Boltzmann y cualquiera para 3 y 4 (más Boltzmann que la otra) con mutación
        uniforme con alta probabilidad de mutación. Boltzmann empieza con exploración, termina con explotación, la combinación de sel 1 
        con sel 2 y mutación nos asegura una requete exploración siempre. Boltzmann hace que cuando la temperatura sea baja, los hijos
        deformes altamente mutados mueran siempre
"""
