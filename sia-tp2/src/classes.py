from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Tuple, Set
import math

# Fuerza, Agilidad, Pericia, Resistencia, Vida, Height
Chromosome = Tuple[float, float, float, float, float, float]
PROPERTIES = ("strength", "agility", "dexterity", "resistance", "health", "height")
PROPERTIES_SUM = 150
MIN_HEIGHT = 1.3
MAX_HEIGHT = 2.0

class BaseClass(ABC):

    def __init__(self, strength: float, agility: float, dexterity: float, resistance: float, health: float,
                 height: float):
        if not (MIN_HEIGHT <= height <= MAX_HEIGHT):
            height = MIN_HEIGHT + ((height - MIN_HEIGHT) % (MAX_HEIGHT - MIN_HEIGHT))

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
        return (self.__agility_p() + self.__dexterity_p()) * self.__strength_p() * self.atm

    def _defense(self):
        return (self.__resistance_p() + self.__dexterity_p()) * self.__health_p() * self.dem

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
        if (self.genes['strength'] + self.genes['agility'] + self.genes['dexterity'] +
                self.genes['resistance'] + self.genes['health']) != PROPERTIES_SUM:
            raise Exception(f"Properties do not sum {PROPERTIES_SUM}")

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


class Guerrero(BaseClass):
    def __init__(self, strength: float, agility: float, dexterity: float, resistance: float, health: float,
                 height: float):
        super().__init__(strength, agility, dexterity, resistance, health, height)

    def get_fitness(self):
        return 0.6 * self._attack() + 0.4 * self._defense()


class Arquero(BaseClass):
    def __init__(self, strength: float, agility: float, dexterity: float, resistance: float, health: float,
                 height: float):
        super().__init__(strength, agility, dexterity, resistance, health, height)

    def get_fitness(self):
        return 0.9 * self._attack() + 0.1 * self._defense()


class Defensor(BaseClass):
    def __init__(self, strength: float, agility: float, dexterity: float, resistance: float, health: float,
                 height: float):
        super().__init__(strength, agility, dexterity, resistance, health, height)

    def get_fitness(self):
        return 0.1 * self._attack() + 0.9 * self._defense()


class Infiltrado(BaseClass):

    def __init__(self, strength: float, agility: float, dexterity: float, resistance: float, health: float,
                 height: float):
        super().__init__(strength, agility, dexterity, resistance, health, height)

    def get_fitness(self):
        return 0.8 * self._attack() + 0.3 * self._defense()
