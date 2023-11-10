from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

letters_order = ['`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~','DEL']

def plot_latent_space(autoencoder: Autoencoder, font):
    x_coord = []
    y_coord = []
    letters = []
    for i, letter in enumerate(font):
        latent_space_xy = autoencoder.run_input(letter)[1]
        x_coord.append(latent_space_xy[0])
        y_coord.append(latent_space_xy[1])
        letters.append(letters_order[i])
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=x_coord, y=y_coord,
                             mode='markers',
                             name='Latent Space',
                             text=letters))

    fig.show()
