import random
from classes import BaseClass, Chromosome, PROPERTIES_SUM, PROPERTIES
from typing import Callable, List
from global_config import config

MutationFunction = Callable[[BaseClass], BaseClass]

mutation_config = config['mutation']

MUTATION_PROBABILITY = mutation_config['mutation_probability']

GENE_VARIATION_BOUNDS = mutation_config['gene_bounds']


def gen_mutation(character: BaseClass) -> BaseClass:
    if random.uniform(0, 1) < MUTATION_PROBABILITY:
        gene_to_mutate = random.sample(PROPERTIES, 1)
        new_gene_value = mutate(character.genes[gene_to_mutate],
                                GENE_VARIATION_BOUNDS[gene_to_mutate]['lower_bound'],
                                GENE_VARIATION_BOUNDS[gene_to_mutate]['upper_bound'])
        character.genes[gene_to_mutate] = new_gene_value
    return character


def mutate_population(population: [BaseClass]) -> List[BaseClass]:
    new_population: List[BaseClass] = []
    for pop in population:
        mutated = mutation_function(pop)
        mutated.apply_bounds()
        new_population.append(mutated)
    return new_population


def mutate(gene: float, lower_bound: float, upper_bound: float) -> float:
    variation = random.uniform(lower_bound, upper_bound)
    return max((gene + variation), 0)


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


def get_mutation_function(string: str) -> MutationFunction:
    if string == 'gen':
        return gen_mutation
    elif string == 'uniform':
        return uniform_multigen_mutation
    elif string == 'limited':
        return limited_multigen_mutation
    elif string == 'complete':
        return complete_mutation
    else:
        raise Exception('Invalid Selection Function Name!')


mutation_function = get_mutation_function(mutation_config['function'])
