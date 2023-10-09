import numpy as np
import plotly.express as px
import pandas as pd
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
    for country in data:
        winner, distance = kohonen.get_most_similar_neuron(country[1])
        print(f"{country[0]} - Winner: ({winner[0]}, {winner[1]}) - Distance: {distance}")





if __name__ == '__main__':
    heatmap_winner_neurons()
