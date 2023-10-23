import json
import random
from sys import argv
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from src.oja import oja
from src.standardization import z_score

CSV_FILE = '../data/europe.csv'


def plot_error(weights_in_epoch, pca_components, epochs, learning_rate):
    error = []
    for w in weights_in_epoch:
        direction_independent_error = min((np.linalg.norm(w[0] - pca_components) ** 2) / len(columns),
                                          (np.linalg.norm(w[0] + pca_components) ** 2) / len(columns))
        error.append([direction_independent_error, w[1]])
    df = pd.DataFrame(error, columns=['ECM', 'Epoch'])
    print(df)
    fig = px.line(df, x='Epoch', y='ECM',
                  title=f"Error between the output of Oja and the Value calculated through the eigenvector<br>"
                        f"<sup>Epochs: {epochs} - Learning Rate: {learning_rate}</sup>",
                  markers='lines+markers')
    fig.update_layout(yaxis_type="log")  # Set y-axis to logarithmic scale
    fig.show()


def plot_pca(pca_from_countries, data_names):
    df = pd.DataFrame({'Country': data_names, 'PC1': pca_from_countries[:, 0]})
    fig = px.bar(data_frame=df, x='Country', y='PC1', text_auto='.3f', title='PC1 per country with PCA Library')
    fig.update_layout(showlegend=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


def plot_oja(final_weights, data, countries, learning_rate, epochs):
    pca_oja = []
    for i, data_row in enumerate(data):
        pca_oja.append([countries[i], np.dot(final_weights, data_row)])
    df = pd.DataFrame(pca_oja, columns=['Country', 'PC1'])
    fig = px.bar(data_frame=df, x='Country', y='PC1', text_auto='.3f',
                 title=f"PC1 per country determined by Oja<br>"
                       f"<sup>Epochs: {epochs} - Learning Rate: {learning_rate}</sup>")
    fig.update_layout(showlegend=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


def plot_ecm_different_eta(data, columns, components, repetitions, etas: List[float], max_epoch):
    random.seed()
    np.random.seed()
    output_data = []
    for eta in etas:
        print(f"ETA: {eta}")
        error = []
        for i in range(0, repetitions):
            print(f"\rREPETITION: {i}", end='')
            weights = oja(data, len(columns), eta, max_epoch)[-1][0]
            error.append(
                (min(np.linalg.norm(weights - components), np.linalg.norm(weights + components)) ** 2) / len(columns)
            )
        print()
        mean = np.mean(error)
        output_data.append([str(eta), mean, mean-np.min(error), np.max(error)-mean])
    df = pd.DataFrame(output_data, columns=['Eta', 'ECM', 'MEAN-MIN', 'MAX-MEAN'])
    print(df)

    title = (f"Error between the output of Oja and the Value from SciKit with different learning rates "
             f"<br><sup>Repetitions: {repetitions}, Max Epochs: {max_epoch}</sup>")

    fig = px.bar(data_frame=df, y='ECM', x='Eta', error_y='MAX-MEAN', error_y_minus='MEAN-MIN', title=title)
    fig.update_layout(showlegend=False,
                      yaxis_type="log")
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


if __name__ == '__main__':
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(data_array)
    if len(argv) < 2:
        print("SEED: None")
        print(f"np.random.seed = {np.random.get_state()[1][0]}")
        print(f"random.seed = {random.getstate()[1][0]}")

        weights_in_epoch = oja(data_array, len(columns))
        plot_pca(pca_features, countries)
        plot_oja(weights_in_epoch[-1][0], data_array, countries, 0.17, 100)
        plot_error(weights_in_epoch, pca.components_)
        plot_ecm_different_eta(data_array, columns, pca.components_, 5,
                               [0.17, 0.125, 0.1, 0.087, 0.05, 0.01, 0.005, 0.001], 150)
    else:
        config = json.load(open(argv[1]))
        if config['random_seed']:
            random.seed(config['random_seed'])
            np.random.seed(config['numpy_random_seed'])
        else:
            print("SEED: None")
            print(f"np.random.seed = {np.random.get_state()[1][0]}")
            print(f"random.seed = {random.getstate()[1][0]}")

        weights_in_epoch = oja(data_array, len(columns), config['learning_rate'], config['max_epochs'])
        plot_pca(pca_features, countries)
        plot_oja(weights_in_epoch[-1][0], data_array, countries, config['learning_rate'], config['max_epochs'])
        plot_error(weights_in_epoch, pca.components_, config['max_epochs'], config['learning_rate'])
        plot_ecm_different_eta(data_array, columns, pca.components_, config['repetitions'],
                               config['test_rates'], config['max_epochs'])
