import numpy as np
import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from src.kohonen import *
import csv

CSV_FILE = '../data/europe.csv'
HEADER_COLUMNS_COUNT = 1  # Tengo que excluir el título del país
GRID_SIZE = 6
MAX_ITERATIONS = 1000
INITIAL_RADIUS = np.sqrt(2)
RADIUS_CHANGE = lambda prev, epoch: prev
LEARNING_RATE = lambda epoch: 0.3


def heatmap_winner_neurons():
    csvfile = open(CSV_FILE)
    csvreader = csv.reader(csvfile)
    header = next(csvreader)

    data = []
    for row in csvreader:
        values = [float(val) for val in row[HEADER_COLUMNS_COUNT:]]
        data.append((row[0], np.array(values, dtype=float)))

    kohonen = Kohonen(k=GRID_SIZE,
                      input_size=len(data[0][1]),
                      max_iterations=MAX_ITERATIONS,
                      get_learning_rate=LEARNING_RATE,
                      distance_function=euclidean_distance,
                      initial_radius=INITIAL_RADIUS,
                      radius_change=RADIUS_CHANGE,
                      data=list(map(lambda x: x[1], data)),
                      random_initial_weights=True)

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
        countries_names_foreach_neuron[winner[0]][winner[1]] += f"{country[0]}, "
        if countries_count_foreach_neuron[winner[0]][winner[1]] != 0 and countries_count_foreach_neuron[winner[0]][winner[1]] % 3 == 0:
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
