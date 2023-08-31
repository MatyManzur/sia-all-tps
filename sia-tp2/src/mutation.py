import copy
import random

from classes import BaseClass, Chromosome, PROPERTIES_SUM, PROPERTIES
from typing import Callable

MutationFunction = Callable[[BaseClass], BaseClass]

MUTATION_PROBABILITY = 0.2  # traer de config

GENE_VARIATION_BOUNDS = { # traer de config
    "strength": {
        "lower_bound": -10,
        "upper_bound": 10,
    },
    "agility": {
        "lower_bound": -10,
        "upper_bound": 10,
    },
    "dexterity": {
        "lower_bound": -10,
        "upper_bound": 10,
    },
    "resistance": {
        "lower_bound": -10,
        "upper_bound": 10,
    },
    "health": {
        "lower_bound": -10,
        "upper_bound": 10,
    },
    "height": {
        "lower_bound": -0.5,
        "upper_bound": 0.5,
    },
}


def mutate(gene: float, lower_bound: float, upper_bound: float) -> float:
    variation = random.uniform(lower_bound, upper_bound)
    return max((gene + variation), 0)


def gen_mutation(character: BaseClass) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        gene_to_mutate = random.sample(PROPERTIES, 1)
        new_gene_value = mutate(character.genes[gene_to_mutate],
                                GENE_VARIATION_BOUNDS[gene_to_mutate]['lower_bound'],
                                GENE_VARIATION_BOUNDS[gene_to_mutate]['upper_bound'])
        character.genes[gene_to_mutate] = new_gene_value
    return character


def limited_multigen_mutation(character: BaseClass) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        gene_count = random.randint(0, len(PROPERTIES))
        genes_to_mutate = random.sample(PROPERTIES, gene_count)
        for gene in genes_to_mutate:
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character


def uniform_multigen_mutation(character: BaseClass) -> BaseClass:
    for gene in PROPERTIES:
        if random.uniform(0, 1) < MUTATION_PROBABILITY:
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character


def complete_mutation(character: BaseClass) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        for gene in PROPERTIES:
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character
