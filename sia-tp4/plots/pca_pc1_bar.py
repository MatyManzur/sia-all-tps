import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import pycountry as pyc

def pc1_bar():
    dataset = pd.read_csv('../data/europe.csv')
    columns = dataset.columns
    features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

    x_scaled = StandardScaler().fit_transform(dataset[features])

    pca = PCA(n_components=1)
    pca_features = pca.fit_transform(x_scaled)  # paises en la nueva base de componentes pcpales

    df = pd.DataFrame(data=list(map(lambda x: x[0], pca_features)), columns=['PC1'], index=dataset['Country'])

    fig = px.bar(data_frame=df, text_auto='.2f')
    fig.update_layout(yaxis_title='PCA1', title='PCA1 per country', showlegend=False)
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.show()

if __name__ == '__main__':
    pc1_bar()