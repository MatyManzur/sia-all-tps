import numpy as np

from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

LETTERS = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']


def plot_latent_space(autoencoder: Autoencoder, font, font_names=LETTERS, chosen_font=LETTERS[1], font_shape=[7,5], should_round=True):
    x_coord = []
    y_coord = []
    letters = []
    fig2 = make_subplots(rows=4, cols=8)
    chosen_letter_to_compare = chosen_font
    letter_index = font_names.index(chosen_letter_to_compare)
    other_ls_xy = autoencoder.run_input(font[letter_index])[1]
    for i, letter in enumerate(font):
        latent_space_xy = autoencoder.run_input(letter)[1]
        x_coord.append(latent_space_xy[0][0])
        y_coord.append(latent_space_xy[1][0])
        letters.append(font_names[i])
        # Generamos la letra a la mitad entre esta y la letra elegida
        weird_letter_ls_xy = (other_ls_xy + latent_space_xy) / 2
        weird_letter = autoencoder.output_from_latent_space(
            (weird_letter_ls_xy[0], weird_letter_ls_xy[1])
        )
        weird_letter = np.reshape(round(weird_letter) if should_round else weird_letter, font_shape)
        fig2.add_trace(go.Heatmap(
            z=np.flipud(weird_letter), colorscale=[[0, 'white'], [1, 'black']]),
            row=1 + i // 8,
            col=1 + i % 8)

    fig = go.Figure()

    fig = px.scatter(x=x_coord, y=y_coord, text=letters)
    fig.update_traces(textposition='top center')

    fig.update_layout(height=800, title_text='Latent Space')

    fig.show()
    fig2.show()
