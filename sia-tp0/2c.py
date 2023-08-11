import json
import sys
import plotly.express as px
import pandas as pd
import numpy as np
from src.pokemon import PokemonFactory, StatusEffect
from pokemon_aux import generate_pokemon_with_state, average_catch_rate

if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        base_name = config["pokemon"]
        pokeballs = config["pokeball"]
        data_status = []
        data_hp = []
        data_pb = []
        pokemon_pb = generate_pokemon_with_state(base_name, 100, StatusEffect.NONE, 1)
        for hp in range(10):
            data_hp.append(average_catch_rate(
                generate_pokemon_with_state(base_name, 100, StatusEffect.NONE, hp * 0.1),
                pokeballs[0], 1000))
        for status in StatusEffect:
            data_status.append(
                average_catch_rate(generate_pokemon_with_state(base_name, 100, status, 1), pokeballs[0], 1000))
        for pb in pokeballs:
            data_pb.append(average_catch_rate(pokemon_pb, pb, 1000))
        print(data_status)

        data_var = [np.var(data_status), np.var(data_hp), np.var(data_pb)]
        df = pd.DataFrame(data_var,index=['Status Effect','HP','Pokeball'],columns=['Variance'])
        px.bar(df,x=df.index,y=df.columns).show()

