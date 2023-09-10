import random
from classes import BaseClass, Chromosome, PROPERTIES_SUM, PROPERTIES
from typing import Callable, List
from global_config import config
import numpy as np

MutationFunction = Callable[[BaseClass,int], BaseClass]
NonUniformMutationProbability = Callable[[int], float]

mutation_config = config['mutation']

MUTATION_PROBABILITY = mutation_config['mutation_probability']

GENE_VARIATION_BOUNDS = mutation_config['gene_bounds']

NON_UNIFORM_TYPE = mutation_config['non_uniform_type']

NON_UNIFORM_SPEED = mutation_config['non_uniform_speed']


def non_uniform_mutation_probability(generation: int) -> float:
    if NON_UNIFORM_TYPE == 'increasing':
        return 1 - np.exp(-generation / NON_UNIFORM_SPEED)
    elif NON_UNIFORM_TYPE == 'decreasing':
        return np.exp(-generation / NON_UNIFORM_SPEED)
    elif NON_UNIFORM_TYPE == 'sinusoidal':
        return 0.5 + 0.5 * np.sin(generation/NON_UNIFORM_SPEED)
    else:
        raise Exception('Invalid Non Uniform Type!')


def gen_mutation(character: BaseClass,generation:int) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        gene_to_mutate = random.sample(PROPERTIES, 1)[0]
        new_gene_value = mutate(character.genes[gene_to_mutate],
                                GENE_VARIATION_BOUNDS[gene_to_mutate]['lower_bound'],
                                GENE_VARIATION_BOUNDS[gene_to_mutate]['upper_bound'])
        character.genes[gene_to_mutate] = new_gene_value
    return character


def mutate_population(population: [BaseClass],generation:int) -> List[BaseClass]:
    new_population: List[BaseClass] = []
    for pop in population:
        mutated = mutation_function(pop,generation)
        mutated.apply_bounds()
        new_population.append(mutated)
    return new_population


def mutate(gene: float, lower_bound: float, upper_bound: float) -> float:
    variation = random.uniform(lower_bound, upper_bound)
    return max((gene + variation), 0)


def limited_multigen_mutation(character: BaseClass,generation:int) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        gene_count = random.randint(1, len(PROPERTIES))
        genes_to_mutate = random.sample(PROPERTIES, gene_count)
        for gene in genes_to_mutate:
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character


def uniform_multigen_mutation(character: BaseClass,generation:int) -> BaseClass:
    for gene in PROPERTIES:
        if random.uniform(0, 1) < MUTATION_PROBABILITY:
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character


def non_uniform_multigen_mutation(character: BaseClass,generation:int) -> BaseClass:
    for gene in PROPERTIES:
        if random.uniform(0, 1) < non_uniform_mutation_probability(generation):
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character


def complete_mutation(character: BaseClass,generation:int) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        for gene in PROPERTIES:
            new_gene_value = mutate(character.genes[gene],
                                    GENE_VARIATION_BOUNDS[gene]['lower_bound'],
                                    GENE_VARIATION_BOUNDS[gene]['upper_bound'])
            character.genes[gene] = new_gene_value
    return character


def get_mutation_function(string: str) -> MutationFunction:
    if string == 'gen':
        return gen_mutation
    elif string == 'uniform':
        return uniform_multigen_mutation
    elif string == 'limited':
        return limited_multigen_mutation
    elif string == 'complete':
        return complete_mutation
    elif string == 'non_uniform':
        return non_uniform_multigen_mutation
    else:
        raise Exception('Invalid Selection Function Name!')


mutation_function = get_mutation_function(mutation_config['function'])
