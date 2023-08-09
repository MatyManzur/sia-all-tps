import json
import sys

import numpy as np
import plotly.express as px
from src.catching import attempt_catch
from src.pokemon import PokemonFactory, StatusEffect


if __name__ == "__main__":
    factory = PokemonFactory("pokemon.json")
    # Ejemplo clase
    # with open(f"{sys.argv[1]}", "r") as f:
    #     config = json.load(f)
    #     ball = config["pokeball"]
    #     pokemon = factory.create(config["pokemon"], 100, StatusEffect.NONE, 1)
    #     print("No noise: ", attempt_catch(pokemon, config["pokeball"]))
    #     for _ in range(10):
    #         print("Noisy: ", attempt_catch(pokemon, config["pokeball"], 0.15))

    # Pregunta 1a)
    pokemonsf = open("pokemon.json", "r")
    pokemons = list(json.load(pokemonsf).keys())
    pokeballs = ["pokeball", "ultraball", "fastball", "heavyball"]
    results = {}
    for ball in pokeballs:
        ball_rates = {}
        for pokemon in pokemons:
            # print(f"Pokemon: {pokemon}, Pokeball: {ball}")
            bicho = factory.create(pokemon, 100, StatusEffect.NONE, 1)
            ans = []
            for _ in range(100):
                ans.append(attempt_catch(bicho, ball, 0)[0])
            # Tengo un array con 100 True/False, si le calculo el promedio es hacer sucess/total_attempts
            # print(np.average(ans))
            average = np.average(ans)
            ball_rates[pokemon] = average
        # print(f"Ball type: {ball}, Success Rate: {np.average(ball_rates)}")
        results[ball] = ball_rates

    fig = px.bar(x=list(results.keys()), y=list(map(lambda v: np.average(list(v.values())), results.values())), title="Average capture probability by Pokeball")
    fig.show()

    # Pregunta 1b)


