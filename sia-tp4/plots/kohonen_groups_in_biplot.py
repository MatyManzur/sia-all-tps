import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.kohonen import *
from src.standardization import z_score
from sys import argv

CSV_FILE = '../data/europe.csv'
HEADER_COLUMNS_COUNT = 1  # Tengo que excluir el título del país
GRID_SIZE = 3
MAX_ITERATIONS = 10000
INITIAL_RADIUS = 3
SEED = 885  # 703 para grid de 4 y 885 para grid de 3
RADIUS_CHANGE = lambda prev, epoch: max(INITIAL_RADIUS - 0.05 * epoch, 1)
LEARNING_RATE = lambda epoch: 0.1 * (1.0 - (epoch / MAX_ITERATIONS))
INITIALIZE_RANDOM_WEIGHTS = False


def kohonen_pca_clusters(kohonen: Kohonen, data_array: List, countries: List, grid_size: int):
    color_grid = [
        ["midnightblue", "green", "red"],
        ["blue", "aquamarine", "orange"],
        ["lightskyblue", "darkseagreen", "yellow"]
    ]
    colors = []
    for i, country in enumerate(countries):
        winner, distance = kohonen.get_most_similar_neuron(data_array[i])
        colors.append(color_grid[winner[0]][winner[1]])

    dataset = pd.read_csv('../data/europe.csv')
    columns = dataset.columns
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

    x_scaled = StandardScaler().fit_transform(dataset[features])

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(x_scaled)  # paises en la nueva base de componentes pcpales

    print(pca.components_)  # array de autovectores

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # matriz de las cargas
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter(pca_features, x=0, y=1, text=dataset['Country'],
                     title=f'PCA BiPlot - Total Explained Variance {total_var:.2f}%')

    fig.update_traces(marker_color="rgba(0,0,0,0)")
    fig.update_traces(textposition='bottom right')
    fig.update_layout(xaxis_title='PCA1', yaxis_title='PCA2', showlegend=False)

    for i, row in enumerate(pca_features):
        fig.add_scatter(
            x=[row[0]],
            y=[row[1]],
            marker=dict(
                color=colors[i],
                size=10
            )
        )

    fig.show()

    return 0


if __name__ == '__main__':
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
                      seed=SEED,
                      random_initial_weights=INITIALIZE_RANDOM_WEIGHTS)
    kohonen.train()
    kohonen_pca_clusters(kohonen, data_array, countries, GRID_SIZE)

