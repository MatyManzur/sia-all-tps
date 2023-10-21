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
MAX_ITERATIONS = 10000
RADIUS_CHANGE = lambda prev, epoch: max(prev - 0.05 * epoch, 1)
LEARNING_RATE = lambda epoch: 0.1 * (1.0 - (epoch / MAX_ITERATIONS))


def get_dead_neurons(data, grid_size: int, random_weights: bool, seed: int):
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    kohonen = Kohonen(k=grid_size,
                      input_size=len(columns),
                      max_iterations=MAX_ITERATIONS,
                      get_learning_rate=LEARNING_RATE,
                      distance_function=euclidean_distance,
                      initial_radius=grid_size * 0.7,
                      radius_change=RADIUS_CHANGE,
                      standardized_data=list(data_array),
                      seed=seed,
                      random_initial_weights=random_weights)
    kohonen.train()
    countries_count_foreach_neuron = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for i, country in enumerate(countries):
        winner, distance = kohonen.get_most_similar_neuron(data_array[i])
        countries_count_foreach_neuron[winner[1]][winner[0]] += 1

    count = 0
    for row in countries_count_foreach_neuron:
        for col in row:
            if col == 0:
                count += 1
    return count


if __name__ == '__main__':
    data = pd.read_csv(CSV_FILE)
    results = {
        "grid_size": [],
        "random_initial_weights": [],
        "average_dead_neurons": [],
        "mean-min_dead_neurons": [],
        "max-mean_dead_neurons": []
    }
    SAMPLE_COUNT = 16
    for size in range(2, 7):
        print(f"-------- GRID SIZE: {size} --------")
        for random_weights in [True, False]:
            dead_neurons = [get_dead_neurons(data, size, random_weights, size*SAMPLE_COUNT + i) for i in range(SAMPLE_COUNT)]
            results['grid_size'].append(size)
            results['random_initial_weights'].append(random_weights)
            results['average_dead_neurons'].append(np.mean(dead_neurons))
            results['mean-min_dead_neurons'].append(np.mean(dead_neurons) - np.min(dead_neurons))
            results['max-mean_dead_neurons'].append(np.max(dead_neurons) - np.mean(dead_neurons))
    df = pd.DataFrame(results)
    fig = px.bar(
        data_frame=df,
        x='grid_size',
        y='average_dead_neurons',
        color='random_initial_weights',
        barmode='group',
        error_y='max-mean_dead_neurons',
        error_y_minus='mean-min_dead_neurons'
    )
    fig.show()
