import math
from typing import List, Callable

from global_config import get_config,config
from classes import BaseClass, Chromosome, chromosome_equals

finish_config = config["finish_criteria"]

FinishFunction = Callable[[List[BaseClass], int, float], bool]


def change_finish_config():
    global finish_config
    finish_config = get_config()["finish_criteria"]


# in miliseconds
def time_based(population: List[BaseClass], generation: int, execution_time: float) -> bool:
    return execution_time > finish_config["finish_time_ms"]


# similarity is defined as delta of genomes
def structure_based_init() -> Callable[[List[BaseClass], int, float], bool]:
    prev_generation: list[Chromosome] = []
    generation_count = 0

    def structure_based(population: List[BaseClass], generation: int, execution_time: float):
        nonlocal prev_generation
        nonlocal generation_count
        matches = get_matches(list(map(BaseClass.get_cromies, population)), prev_generation)

        if matches < finish_config["relevant_population"] * len(population):
            prev_generation = list(map(BaseClass.get_cromies, population))
            return False
        generation_count += 1
        return generation_count >= finish_config["relevant_generations"]

    return structure_based


def content_based_init() -> Callable[[List[BaseClass], int, float], bool]:
    prev_best = -1
    generation_count = 0

    def content_based(population: List[BaseClass], generation: int, execution_time: float):
        nonlocal prev_best
        nonlocal generation_count
        best_fitness = max(population, key=lambda x: x.get_fitness()).get_fitness()
        if prev_best == -1 or not math.isclose(best_fitness, prev_best):
            generation_count = 0
            prev_best = best_fitness
            return False
        generation_count += 1
        return generation_count >= finish_config["relevant_generations"]

    return content_based


def generation_based(population: List[BaseClass], generation: int, execution_time: float) -> bool:
    return generation >= finish_config["finish_generation_count"]


def optimum_based(population: List[BaseClass], generation: int, execution_time: float) -> bool:
    best = max(population, key=lambda x: x.get_fitness())
    return best.get_fitness() >= finish_config["finish_optimum"]["acceptance"]


def get_finish_condition(name: str) -> FinishFunction:
    if name == "time_based":
        return time_based
    elif name == "structure_based":
        return structure_based_init()
    elif name == "content_based":
        return content_based_init()
    elif name == "optimum_based":
        return optimum_based
    elif name == "generation_based":
        return generation_based


def get_matches(population: list[Chromosome], to_match: list[Chromosome]) -> int:
    copy_pop = list(population)
    matches = 0
    for _, chromosome1 in enumerate(to_match):
        for _, chromosome2 in enumerate(copy_pop):
            if chromosome_equals(chromosome1, chromosome2, finish_config["similarity"]):
                matches += 1
                copy_pop.remove(chromosome2)
    return matches
