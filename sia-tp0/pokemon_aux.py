import pandas as pd
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect, Pokemon
from typing import List

def average_catch_rate(
        pokemon_to_catch: Pokemon, pokeball: str, iterations: int
) -> float:
    iterate = range(iterations)
    success_count = 0
    for i in iterate:
        success_count += attempt_catch(pokemon_to_catch, pokeball)[0]
    return success_count/iterations


def generate_pokemons_with_state(
        pokes: List[str], lvl: int, status: StatusEffect, hp: int
) -> List[Pokemon]:
    pokemon_list = []
    factory_aux = PokemonFactory("pokemon.json")
    for p in pokes:
        pokemon_list.append(factory_aux.create(p, lvl, status, hp))
    return pokemon_list


