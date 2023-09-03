from typing import List, Callable

from global_config import config
from classes import Warrior, Rogue, Warden, Archer, BaseClass


# in miliseconds
def time_based(population: List[BaseClass], generation: int, execution_time: float) -> bool:
    return execution_time > config["finish_criteria"]["finish_time_ms"]


def structure_based_init() -> Callable[[List[BaseClass], int, float], bool]:
    prev_generation = set()
    generation_count = 0

    def structure_based(population: List[BaseClass], generation: int, execution_time: float):
        nonlocal prev_generation
        nonlocal generation_count
        matches = get_matches(population, prev_generation)
        prev_generation = set(map(BaseClass.get_fitness, population))

        if len(prev_generation) == 0 or matches < config["finish_criteria"]["relevant_population"] * len(population):
            prev_generation = set(map(BaseClass.get_fitness, population))
            return False
        generation_count += 1
        return generation_count >= config["finish_criteria"]["relevant_generations"]

    return structure_based


def content_based_init() -> Callable[[List[BaseClass], int, float], bool]:
    prev_best = -1
    generation_count = 0

    def content_based(population: List[BaseClass], generation: int, execution_time: float):
        nonlocal prev_best
        nonlocal generation_count
        best_fitness = max(population, key=lambda x: x.get_fitness()).get_fitness()
        if prev_best == -1 or best_fitness > prev_best:
            generation_count = 0
            prev_best = best_fitness
            return False
        generation_count += 1
        return generation_count >= config["finish_criteria"]["relevant_generations"]

    return content_based


def generation_based(population: List[BaseClass], generation: int, execution_time: float) -> bool:
    return generation >= config["finish_criteria"]["finish_generation_count"]


def optimum_based(population: List[BaseClass], generation: int, execution_time: float) -> bool:
    best = max(population, key=lambda x: x.get_fitness())
    return best.get_fitness() >= config["finish_criteria"]["finish_optimum"]["acceptance"]


def get_matches(population: List[BaseClass], to_match: set[float]) -> int:
    return len(to_match.intersection(set(map(BaseClass.get_fitness, population))))
