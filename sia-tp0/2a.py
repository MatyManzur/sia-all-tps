import json
import sys
import plotly.express as px
import pandas as pd
from src.pokemon import PokemonFactory, StatusEffect
from src.catching import attempt_catch
from pokemon_aux import generate_pokemon_with_state,average_catch_rate


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        pokemons = []
        base_name = config["pokemon"]
        base_pokeball = config["pokeball"]
        for status in StatusEffect:
            pokemons.append(generate_pokemon_with_state(base_name,100,status,1))
        status_data = []
        for pk in pokemons:
            status_data.append([pk.status_effect.name,average_catch_rate(pk,base_pokeball,1000)])
        df = pd.DataFrame(status_data,columns=["Status Effect","Catch Rate"])
        px.bar(df,x='Status Effect',y='Catch Rate',title=f'Catch Rate of {base_name} with {base_pokeball} depending of Status Effect').show()