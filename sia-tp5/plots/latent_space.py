from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

letters_order = ['`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~','DEL']

def plot_latent_space(autoencoder: Autoencoder, font):
    x_coord = []
    y_coord = []
    letters = []
    for i, letter in enumerate(font):
        latent_space_xy = autoencoder.run_input(letter)[1]
        x_coord.append(latent_space_xy[0][0])
        y_coord.append(latent_space_xy[1][0])
        letters.append(letters_order[i])
    fig = go.Figure()

    fig = px.scatter(x=x_coord, y=y_coord, text=letters)
    fig.update_traces(textposition='top center')
    
    fig.update_layout(height=800,title_text='Latent Space')

    fig.show()
