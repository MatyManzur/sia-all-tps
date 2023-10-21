import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from src.kohonen import *
from src.standardization import z_score
from sys import argv

CSV_FILE = '../data/europe.csv'
HEADER_COLUMNS_COUNT = 1  # Tengo que excluir el título del país
GRID_SIZE = 2
MAX_ITERATIONS = 10000
INITIAL_RADIUS = 2
SEED = 11  # 5 para grid de 4 y 11 para grid de 3
RADIUS_CHANGE = lambda prev, epoch: max(INITIAL_RADIUS - 0.05 * epoch, 1)
LEARNING_RATE = lambda epoch: 0.1 * (1.0 - (epoch / MAX_ITERATIONS))
INITIALIZE_RANDOM_WEIGHTS = False


def get_unified_mean_distance(weight_matrix: NDArray[float]):
    neighborhood_radius = 1
    rows, cols, _ = weight_matrix.shape
    u_matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            neuron = weight_matrix[i, j]
            i_min = max(0, i - neighborhood_radius)
            i_max = min(rows, i + neighborhood_radius + 1)
            j_min = max(0, j - neighborhood_radius)
            j_max = min(cols, j + neighborhood_radius + 1)
            neighborhood_arr = weight_matrix[i_min:i_max, j_min:j_max]
            distance = np.linalg.norm(neighborhood_arr - neuron, axis = -1)
            u_matrix[i, j] = np.mean(distance)
    return u_matrix


def heatmap_winner_neurons(grid_size: int, max_iterations: int, initial_radius: float,
                           radius_change: Callable[[float, int], float],
                           learning_rate: Callable[[int], float], initialize_random_weights: bool):
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()
    for i, country in enumerate(countries):
        print(f"{country}: {data_array[i]}")

    print("=" * 20)

    kohonen = Kohonen(k=grid_size,
                      input_size=len(columns),
                      max_iterations=max_iterations,
                      get_learning_rate=learning_rate,
                      distance_function=euclidean_distance,
                      initial_radius=initial_radius,
                      radius_change=radius_change,
                      standardized_data=list(data_array),
                      seed=SEED,
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

        countries_names_foreach_neuron[winner[1]][winner[0]] += f", {country}" if \
        countries_names_foreach_neuron[winner[1]][winner[0]] != '' and countries_count_foreach_neuron[winner[1]][
            winner[0]] % 3 != 0 else f"{country}"

        countries_count_foreach_neuron[winner[1]][winner[0]] += 1
        if countries_count_foreach_neuron[winner[1]][winner[0]] != 0 and countries_count_foreach_neuron[winner[1]][
            winner[0]] % 3 == 0:
            countries_names_foreach_neuron[winner[1]][winner[0]] += "<br>"

        countries_winners["countries"].append(country)
        countries_winners["winner_row"].append(winner[1])
        countries_winners["winner_col"].append(winner[0])

        print(f"{country} - Winner: ({winner[0]}, {winner[1]}) - Distance: {distance}")

    distance = get_unified_mean_distance(kohonen.weights)

    distance_heatmap = go.Heatmap(z=distance, text=countries_names_foreach_neuron, texttemplate="%{text}", colorscale='oranges')

    heatmap = go.Heatmap(z=countries_count_foreach_neuron,
                         text=countries_names_foreach_neuron,
                         texttemplate="%{text}",
                         colorscale='greys'
                         )

    variable_heatmaps = []
    variable_layouts = []

    for variable in columns:
        weights = kohonen.weights[:, :, columns.index(variable)]
        print(f"Variable: {variable} - Weights: {weights}")
        variable_heatmaps.append(go.Heatmap(z=weights, text=countries_names_foreach_neuron,
                                            texttemplate="%{text}",
                                            colorscale='RdBu', zmid=0))
        variable_layouts.append(go.Layout(title=f"Variable per Neuron: {variable}",
                                          xaxis=dict(visible=False), yaxis=dict(visible=False)))


    # Create a layout for the heatmap
    layout = go.Layout(
        title='Countries per Neuron',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    layout2 = go.Layout(
        title="Unified Distance Matrix",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    # Create a figure and add the heatmap trace to it
    fig = go.Figure(data=[heatmap], layout=layout)

    # Show the heatmap
    fig2 = go.Figure(data=[distance_heatmap], layout=layout2)



    # Color de cada grupo
    color_grid = [
        ["midnightblue", "green", "red"],
        ["blue", "aquamarine", "orange"],
        ["lightskyblue", "darkseagreen", "yellow"]
    ]
    for i in range(3):
        for j in range(3):
            fig.add_scatter(x=[j], y=[i-0.3], marker=dict(color=color_grid[i][j], size=30))
    fig.update_layout(showlegend=False)

    fig.show()
    fig2.show()

    for i in range(len(variable_heatmaps)):
        fig3 = go.Figure(data=[variable_heatmaps[i]], layout=variable_layouts[i])
        fig3.show()

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
