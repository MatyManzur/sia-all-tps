import numpy as np
from sklearn.decomposition import PCA
from src.standardization import z_score
import pandas as pd
import plotly.express as px
from src.oja import oja

CSV_FILE = '../data/europe.csv'


def plot_error(weights_in_epoch, pca_components):
    error = []
    for w in weights_in_epoch:
        error.append([(np.linalg.norm(w[0] - pca_components) ** 2) / len(columns), w[1]])
    df = pd.DataFrame(error, columns=['ECM', 'Epoch'])
    print(df)
    fig = px.line(df, x='Epoch', y='ECM', title='Error between the output of Oja and the Value calculated through the '
                                                'eigenvector', markers='lines+markers')
    fig.update_layout(yaxis_type="log")  # Set y-axis to logarithmic scale
    fig.show()


def plot_pca(pca_from_countries, data_names):
    df = pd.DataFrame({'Country': data_names, 'PC1': pca_from_countries[:, 0]})
    fig = px.bar(data_frame=df, x='Country', y='PC1', text_auto='.2f',title='PC1 per country with PCA Library')
    fig.update_layout(showlegend=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()
def plot_oja(final_weights, data,countries):
    pca_oja = []
    for i,data_row in enumerate(data):
        pca_oja.append([countries[i], np.dot(final_weights, data_row)])
    df = pd.DataFrame(pca_oja, columns=['Country', 'PC1'])
    fig = px.bar(data_frame=df, x='Country', y='PC1', text_auto='.2f', title='PC1 per country determined by Oja in 100 '
                                                                             'epochs and initial learning rate 0.17')
    fig.update_layout(showlegend=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()

def plot_ecm_different_eta(data,columns,components):
    etas = [0.17, 0.125, 0.1, 0.087, 0.05, 0.01, 0.005, 0.001]
    output_data = []
    for eta in etas:
        error = []
        for i in range(0, 5):
            weights = oja(data, len(columns), eta,150)[-1][0]
            error.append((np.linalg.norm(weights - components) ** 2) / len(columns))
        output_data.append([str(eta), np.mean(error), np.std(error)])
    df = pd.DataFrame(output_data, columns=['Eta', 'ECM', 'STD'])
    print(df)
    fig = px.bar(data_frame=df,y='ECM',x='Eta', error_y='STD', title='Error between the output of Oja and the Value '
                                                                       'from SciKit with different learning rates')
    fig.update_layout(showlegend=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()


if __name__ == '__main__':
    data = pd.read_csv(CSV_FILE)
    columns = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
    data_array = z_score(data[columns].to_numpy())
    countries = data.Country.to_list()

    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(data_array)
    weights_in_epoch = oja(data_array, len(columns))
    # plot_pca(pca_features, countries)
    # plot_oja(weights_in_epoch[-1][0], data_array, countries)
    # plot_error(weights_in_epoch, pca.components_)
    plot_ecm_different_eta(data_array,columns,pca.components_)

