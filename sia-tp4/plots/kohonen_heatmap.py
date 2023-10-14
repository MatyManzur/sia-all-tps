import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go

from src.kohonen import *
import csv

CSV_FILE = '../data/europe.csv'
HEADER_COLUMNS_COUNT = 1  # Tengo que excluir el título del país
GRID_SIZE = 4
MAX_ITERATIONS = 3000
INITIAL_RADIUS = 3.5
RADIUS_CHANGE = lambda prev, epoch: max(INITIAL_RADIUS * (0.96 ** epoch), 1.0)
LEARNING_RATE = lambda epoch: max(0.9 * (0.96 ** epoch), 0.001)
INITIALIZE_RANDOM_WEIGHTS = False


def heatmap_winner_neurons():
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    kohonen = Kohonen(k=GRID_SIZE,
                      input_size=len(columns),
                      max_iterations=MAX_ITERATIONS,
                      get_learning_rate=LEARNING_RATE,
                      distance_function=euclidean_distance,
                      initial_radius=INITIAL_RADIUS,
                      radius_change=RADIUS_CHANGE,
                      standardized_data=list(data_array),
                      random_initial_weights=INITIALIZE_RANDOM_WEIGHTS)

    kohonen.train()
    countries_winners = {}
    countries_count_foreach_neuron = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    countries_names_foreach_neuron = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    # Esto es para el mapa de mapa.py
    countries_winners["countries"] = []
    countries_winners["winner_row"] = []
    countries_winners["winner_col"] = []

    for country in data:
        winner, distance = kohonen.get_most_similar_neuron(country[1])
        countries_count_foreach_neuron[winner[0]][winner[1]] += 1
        countries_names_foreach_neuron[winner[0]][winner[1]] += f"{country}, "
        if countries_count_foreach_neuron[winner[0]][winner[1]] != 0 and countries_count_foreach_neuron[winner[0]][
            winner[1]] % 3 == 0:
            countries_names_foreach_neuron[winner[0]][winner[1]] += "<br>"

        countries_winners["countries"].append(country[0])
        countries_winners["winner_row"].append(winner[0])
        countries_winners["winner_col"].append(winner[1])

        print(f"{country[0]} - Winner: ({winner[0]}, {winner[1]}) - Distance: {distance}")

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


if __name__ == '__main__':
    heatmap_winner_neurons()
