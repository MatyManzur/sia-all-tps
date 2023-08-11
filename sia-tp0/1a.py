import json
import sys
import numpy as np
import plotly.express as px
import pandas as pd
from pokemon_aux import average_catch_rate,generate_pokemons_with_state
from src.pokemon import PokemonFactory, StatusEffect, Pokemon


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        pokemon = generate_pokemons_with_state(config["pokemon"],100,StatusEffect.NONE,1)
        pokeballs = config["pokeball"]
        fig_list = []
        pk = pokemons[0]
        average_list = []
        pk_average_list = []
        for pb in pokeballs:
            for pk in pokemons:
                pk_average_list.append(average_catch_rate(pk, pb, 100))
            average_list.append([pb, np.average(pk_average_list)])
            pk_average_list = []

        (px.bar(pd.DataFrame(average_list,
                             columns=['Pokeball','CatchRate']),x='Pokeball',y='CatchRate',
                title="Average Catch Rate of Pokeballs",
                labels={"x": "Pokeball","y": "Catch Rate"})).show()


