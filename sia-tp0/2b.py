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
        for hp in range(10):
            pokemons.append(generate_pokemon_with_state(base_name,100,StatusEffect.NONE,hp*0.1))
        hp_data = []
        for pk in pokemons:
            hp_data.append([pk.current_hp,average_catch_rate(pk,base_pokeball,1000)])
        df = pd.DataFrame(hp_data,columns=["HP%","Catch Rate"])
        px.line(df,x='HP%',y='Catch Rate',title=f'Catch Rate of {base_name} with {base_pokeball} depending of its HP%',markers=True).show()