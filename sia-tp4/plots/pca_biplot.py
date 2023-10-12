import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import pycountry as pyc


def pca_fun():
    countries_codes = {}
    for country in pyc.countries:
        countries_codes[country.name] = country.alpha_2
    dataset = pd.read_csv('../data/europe.csv')
    columns = dataset.columns
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

    x_scaled = StandardScaler().fit_transform(dataset[features])

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(x_scaled)  # paises en la nueva base de componentes pcpales

    print(pca.components_)  # array de autovectores

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)  # matriz de las cargas
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter(pca_features, x=0, y=1, hover_name=dataset['Country'],
                     title=f'PCA BiPlot - Total Explained Variance {total_var:.2f}%')

    fig.update_traces(marker_color="rgba(0,0,0,0)")
    fig.update_layout(xaxis_title='PCA1', yaxis_title='PCA2')

    for i, row in enumerate(pca_features):
        country_iso = countries_codes[dataset['Country'][i]]
        fig.add_layout_image(
            dict(
                source=f"https://raw.githubusercontent.com/matahombres/CSS-Country-Flags-Rounded/master/flags/{country_iso}.png",
                xref="x",
                yref="y",
                xanchor="center",
                yanchor="middle",
                x=row[0],
                y=row[1],
                sizex=0.4,
                sizey=0.4,
                sizing="contain",
                opacity=1,
                layer="above"
            )
        )

    print(features)
    colors = ['red', 'blue', 'darkgreen', 'darkorange', 'green', 'brown', 'purple']
    for i, feature in enumerate(features):
        fig.add_annotation(
            ax=0, ay=0,
            axref="x", ayref="y",
            x=loadings[i, 0],
            y=loadings[i, 1],
            showarrow=True,
            arrowsize=1,
            arrowhead=1,
            xanchor="right",
            yanchor="top",
            arrowcolor=colors[i]
        )
        fig.add_annotation(
            x=loadings[i, 0],
            y=loadings[i, 1],
            ax=0, ay=0,
            xanchor="center",
            yanchor="bottom",
            text=feature,
            yshift=5,
            font={"size": 15, "color": colors[i]}
        )
    fig.show()


if __name__ == '__main__':
    pca_fun()
