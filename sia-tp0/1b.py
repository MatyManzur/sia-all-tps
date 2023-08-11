import json
import sys
import numpy as np
import plotly.express as px
import pandas as pd
from src.pokemon import PokemonFactory, StatusEffect, Pokemon
from pokemon_aux import generate_pokemons_with_state,average_catch_rate

if __name__ == '__main__':
    factory = PokemonFactory("pokemon.json")
    with open(f"{sys.argv[1]}", "r") as f:
        config = json.load(f)
        pokemons = generate_pokemons_with_state(config["pokemon"],100,StatusEffect.NONE,1)
        pokeballs = config["pokeball"]
        average_catch_rate_data = []
        normalized_catch_rate_data = []
        pokeball_catch_rate = {}
        for pk in pokemons:
            for pb in pokeballs:
                catch_rate = average_catch_rate(pk,pb,10000)
                average_catch_rate_data.append([pb,catch_rate,pk.name])
                if(pb == 'pokeball'):
                    pokeball_catch_rate.update({pk.name: catch_rate})
        for i in range(len(average_catch_rate_data)):
            pb = average_catch_rate_data[i][0]
            catch_rate = average_catch_rate_data[i][1]
            pk = average_catch_rate_data[i][2]
            normalized_catch_rate_data.append([pb,catch_rate/pokeball_catch_rate.get(pk),pk])
        df = pd.DataFrame(normalized_catch_rate_data,columns=['Pokeball','CatchRate Comparison','Pokemon'])
        print(df)
        (px.bar(df,x='Pokemon',y='CatchRate Comparison',barmode='group',color='Pokeball',title='Average Catch Rate compared to a Pokeball')).show()