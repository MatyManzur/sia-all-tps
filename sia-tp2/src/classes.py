from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Tuple, Set
import math

# Fuerza, Agilidad, Pericia, Resistencia, Vida, Height
Chromosome = Tuple[float, float, float, float, float, float]
PROPERTIES = ("strength", "agility", "dexterity", "resistance", "health", "height")
PROPERTIES_SUM = 150.0
MIN_HEIGHT = 1.3
MAX_HEIGHT = 2.0
EPSILON = 10 ** -4


def chromosome_equals(chromosome1: Chromosome, chromosome2: Chromosome, delta: float):
    equals = True
    for i in range(0, len(chromosome1)):
        equals = equals and ((chromosome2[i] == 0 and chromosome1[i] == 0) or abs((chromosome1[i] - chromosome2[i])) / (chromosome1[i] + chromosome2[i]) < delta)
        if not equals:
            return False
    return equals


class BaseClass(ABC):

    def __init__(self, strength: float = None, agility: float = None, dexterity: float = None,
                 resistance: float = None, health: float = None, height: float = None,
                 chromosome: Chromosome = None):
        if chromosome is None and any(p is None for p in [strength, agility, dexterity, resistance, health, height]):
            raise Exception('parameters cannot be None when there\'s no Chromosome')

        if chromosome is not None:
            strength, agility, dexterity, resistance, health, height = chromosome

        if height < MIN_HEIGHT:
            height = MIN_HEIGHT
        elif height > MAX_HEIGHT:
            height = MAX_HEIGHT

        self.genes = {
            "strength": strength,
            "agility": agility,
            "dexterity": dexterity,
            "resistance": resistance,
            "health": health,
            "height": height
        }

        if (strength + agility + dexterity + resistance + health) != PROPERTIES_SUM:
            self.__normalize()

        self.atm = 0.5 - (3 * height - 5) ** 4 + (3 * height - 5) ** 2 + height / 2
        self.dem = 2 + (3 * height - 5) ** 4 - (3 * height - 5) ** 2 + height / 2

    # Abstract Method
    def get_fitness(self):
        raise Exception('Not implemented')

    # ojo hay que usar los p, no los items
    def _attack(self):
        return (self.__agility_p() + self.__dexterity_p()) * self.__strength_p() * self.atm  # 160*1.704 = 273

    def _defense(self):
        return (self.__resistance_p() + self.__dexterity_p()) * self.__health_p() * self.dem  # 160*2.84 = 455

    def __normalize(self):
        current_property_sum = (self.genes['strength'] + self.genes['agility'] +
                                self.genes['dexterity'] + self.genes['resistance'] +
                                self.genes['health'])
        norm_factor = PROPERTIES_SUM / current_property_sum
        self.genes['strength'] *= norm_factor
        self.genes['agility'] *= norm_factor
        self.genes['dexterity'] *= norm_factor
        self.genes['resistance'] *= norm_factor
        self.genes['health'] *= norm_factor
        new_property_sum = self.genes['strength'] + self.genes['agility'] + self.genes['dexterity'] + self.genes[
            'resistance'] + self.genes['health']
        if not math.isclose(new_property_sum, PROPERTIES_SUM):
            raise Exception(f"Properties do not sum {PROPERTIES_SUM}: {new_property_sum}")

    def apply_bounds(self):
        if self.genes['height'] < MIN_HEIGHT:
            self.genes['height'] = MIN_HEIGHT
        elif self.genes['height'] > MAX_HEIGHT:
            self.genes['height'] = MAX_HEIGHT

        if self.genes['strength'] + self.genes['agility'] + self.genes['dexterity'] + self.genes[
            'resistance'] + self.genes['health'] != PROPERTIES_SUM:
            self.__normalize()


    def get_cromies(self) -> Chromosome:
        return (self.genes['strength'], self.genes['agility'], self.genes['dexterity'],
                self.genes['resistance'], self.genes['health'], self.genes['height'])

    def __strength_p(self):
        return 100 * math.tanh(0.01 * self.genes['strength'])

    def __agility_p(self):
        return math.tanh(0.01 * self.genes['agility'])

    def __dexterity_p(self):
        return 0.6 * math.tanh(0.01 * self.genes['dexterity'])

    def __resistance_p(self):
        return math.tanh(0.01 * self.genes['resistance'])

    def __health_p(self):
        return 100 * math.tanh(0.01 * self.genes['health'])

    def __hash__(self) -> int:
        return hash(self.get_cromies())

    def __lt__(self, other: BaseClass) -> bool:
        return self.get_fitness() < other.get_fitness()

    def __str__(self):
        return self.genes.__str__()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class Warrior(BaseClass):
    def __init__(self, strength: float = None, agility: float = None, dexterity: float = None, resistance: float = None,
                 health: float = None, height: float = None, chromosome: Chromosome = None):
        super().__init__(strength, agility, dexterity, resistance, health, height, chromosome)

    def get_fitness(self):
        return 0.6 * self._attack() + 0.4 * self._defense()  # 346


class Archer(BaseClass):
    def __init__(self, strength: float = None, agility: float = None, dexterity: float = None,
                 resistance: float = None, health: float = None, height: float = None,
                 chromosome: Chromosome = None):
        super().__init__(strength, agility, dexterity, resistance, health, height, chromosome)

    def get_fitness(self):
        return 0.9 * self._attack() + 0.1 * self._defense()  # 292


class Warden(BaseClass):
    def __init__(self, strength: float = None, agility: float = None, dexterity: float = None,
                 resistance: float = None, health: float = None, height: float = None,
                 chromosome: Chromosome = None):
        super().__init__(strength, agility, dexterity, resistance, health, height, chromosome)

    def get_fitness(self):
        return 0.1 * self._attack() + 0.9 * self._defense()  # 437


class Rogue(BaseClass):

    def __init__(self, strength: float = None, agility: float = None, dexterity: float = None,
                 resistance: float = None, health: float = None, height: float = None,
                 chromosome: Chromosome = None):
        super().__init__(strength, agility, dexterity, resistance, health, height, chromosome)

    def get_fitness(self):
        return 0.8 * self._attack() + 0.3 * self._defense()  # 355
