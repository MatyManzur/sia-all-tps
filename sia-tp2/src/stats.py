from classes import *
from typing import List, Callable


# Definicion de gene diversity = alelos unicos / total
def get_gene_diversity(population: List[BaseClass]):
  unique = set(item for tup in map(BaseClass.get_cromies, population) for item in tup)
  return len(unique) / (len(population) * 6)

# Para unir stats
# Formato de devolucion es (generation_num, population, best, diversity)
def get_generation_stats(generation: int, population: List[BaseClass]):
  return (
    generation, 
    population, 
    max(population, key=lambda x: x.get_fitness()),
    get_gene_diversity(population)
  )

def print_generation(generation: int, population: List[BaseClass]):
  (_, _, best, gene_diversity) = get_generation_stats(generation, population)
  print(f"{generation} - Best: {best.get_fitness()} - Diversity: {gene_diversity}")