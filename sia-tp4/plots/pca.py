import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px


def pca_fun():
    dataset = pd.read_csv('../data/europe.csv')
    columns = dataset.columns
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

    x_scaled = StandardScaler().fit_transform(dataset[features])

    pca = PCA(n_components=5)
    pca_features = pca.fit_transform(x_scaled) # paises en la nueva base de componentes pcpales

    print(pca.components_) # array de autovectores

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_) # matriz de las cargas
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter(pca_features, x=0, y=1, text=dataset['Country'], title=f'Total Explained Variance {total_var:.2f}%')


    print(features)

    for i, feature in enumerate(features):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0],
            y=loadings[i, 1],
            showarrow=True,
            arrowsize=2,
            arrowhead=2,
            xanchor="right",
            yanchor="top"
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            yshift=5,
        )
    fig.show()



if __name__ == '__main__':
    # csv_array = genfromtxt('../data/europe.csv')
    pca_fun()
