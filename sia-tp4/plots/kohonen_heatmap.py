import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from src.kohonen import *
from src.standardization import z_score
from sys import argv

CSV_FILE = 'data/europe.csv'
HEADER_COLUMNS_COUNT = 1  # Tengo que excluir el título del país
GRID_SIZE = 4
MAX_ITERATIONS = 10000
INITIAL_RADIUS = math.sqrt(2)
RADIUS_CHANGE = lambda prev, epoch: INITIAL_RADIUS - 0.5 * (epoch // 250)
LEARNING_RATE = lambda epoch: 0.1 * (1.0 - (epoch / MAX_ITERATIONS))
INITIALIZE_RANDOM_WEIGHTS = True

# np.random.seed(None)
# random.seed(None)


def heatmap_winner_neurons(grid_size: int, max_iterations: int, initial_radius: float, radius_change: Callable[[float, int], float],
                            learning_rate: Callable[[int], float], initialize_random_weights: bool):
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    kohonen = Kohonen(k=grid_size,
                      input_size=len(columns),
                      max_iterations=max_iterations,
                      get_learning_rate=learning_rate,
                      distance_function=euclidean_distance,
                      initial_radius=initial_radius,
                      radius_change=radius_change,
                      standardized_data=list(data_array),
                      random_initial_weights=initialize_random_weights)

    kohonen.train()
    countries_winners = {}
    countries_count_foreach_neuron = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    countries_names_foreach_neuron = [['' for _ in range(grid_size)] for _ in range(grid_size)]
    # Esto es para el mapa de mapa.py
    countries_winners["countries"] = []
    countries_winners["winner_row"] = []
    countries_winners["winner_col"] = []

    for i, country in enumerate(countries):
        winner, distance = kohonen.get_most_similar_neuron(data_array[i])
        
        countries_names_foreach_neuron[winner[0]][winner[1]] += f", {country}" if countries_names_foreach_neuron[winner[0]][winner[1]]!='' and countries_count_foreach_neuron[winner[0]][winner[1]] % 3 != 0 else f"{country}"

        countries_count_foreach_neuron[winner[0]][winner[1]] += 1
        if countries_count_foreach_neuron[winner[0]][winner[1]] != 0 and countries_count_foreach_neuron[winner[0]][
            winner[1]] % 3 == 0:
            countries_names_foreach_neuron[winner[0]][winner[1]] += "<br>"

        countries_winners["countries"].append(country)
        countries_winners["winner_row"].append(winner[0])
        countries_winners["winner_col"].append(winner[1])

        print(f"{country} - Winner: ({winner[0]}, {winner[1]}) - Distance: {distance}")

    # Create a heatmap trace with text annotations
    heatmap = go.Heatmap(z=countries_count_foreach_neuron, text=countries_names_foreach_neuron, texttemplate="%{text}")

    # Create a layout for the heatmap
    layout = go.Layout(
        title='Heatmap with Text Annotations',
        xaxis=dict(title='X-axis Labels'),
        yaxis=dict(title='Y-axis Labels')
    )

    # Create a figure and add the heatmap trace to it
    fig = go.Figure(data=[heatmap], layout=layout)

    # Show the heatmap
    fig.show()

    return countries_winners

def get_learning_rate(initial, function_name):
    if function_name == 'linear':\
        return lambda epoch: initial * (1.0 - (epoch / MAX_ITERATIONS))
    elif function_name == 'exponential':
        return lambda epoch: initial * math.exp(-epoch / MAX_ITERATIONS)
    elif function_name == 'inverse':
        return lambda epoch: initial / (1.0 + (epoch / MAX_ITERATIONS))
    else:
        return None

def get_radius_change(function_name):
    if function_name == 'linear':
        return lambda prev, epoch: prev - 1
    elif function_name == 'exponential':
        return lambda prev, epoch: prev * math.exp(-epoch / MAX_ITERATIONS)
    elif function_name == 'inverse':
        return lambda prev, epoch: prev / (1.0 + (epoch / MAX_ITERATIONS))
    else:
        return None


if __name__ == '__main__':
    if len(argv) == 1:
        heatmap_winner_neurons(
            grid_size=GRID_SIZE,
            max_iterations=MAX_ITERATIONS,
            initial_radius=INITIAL_RADIUS,
            radius_change=RADIUS_CHANGE,
            learning_rate=LEARNING_RATE,
            initialize_random_weights=INITIALIZE_RANDOM_WEIGHTS
        )
    else:
        config = json.load(open(argv[1]))
        learning_rate = get_learning_rate(config['initial_learning_rate'], config['learning_function'])
        radius_change = get_radius_change(config['radius_change'])
        heatmap_winner_neurons(
            grid_size=config['grid_size'],
            max_iterations=config['max_iterations'],
            initial_radius=config['initial_radius'],
            radius_change=radius_change,
            learning_rate=learning_rate,
            initialize_random_weights=config['initialize_random_weights']
        )
