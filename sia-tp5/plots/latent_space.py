import numpy as np

from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import matplotlib.pyplot as plt
import numpy as np

from src.vae import VariationalAutoencoder

# import mplcursors

LETTERS = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']


def plot_latent_space(autoencoder: Autoencoder | VariationalAutoencoder, font, font_names=LETTERS, chosen_font=LETTERS[1], font_shape=[7,5], should_round=True):
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
    
    plot_interactive_latent_space(autoencoder, x_coord, y_coord, letters, should_round, font_shape)




def plot_interactive_latent_space(autoencoder, x, y, letters, should_round, font_shape):
    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.scatter(x,y)
    for index in range(len(x)):
        plt.text(x[index], y[index] * (1 + 0.03) , letters[index], fontsize=12)

    # Define the update function
    def add_heatmap(event):
        # Get the x and y coordinates of the click
        x_click, y_click = event.xdata, event.ydata
        if x_click is not None and y_click is not None:
            
            weird_letter = autoencoder.output_from_latent_space((x_click, y_click))
            weird_letter = np.reshape(round(weird_letter) if should_round else weird_letter, font_shape)
            

            ptfig = go.Figure(data=go.Heatmap(
                    z=np.flipud(weird_letter), colorscale=[[0, 'white'], [1, 'black']]))
            ptfig.show()

    # Connect the click event to the update function
    fig.canvas.mpl_connect('button_press_event', add_heatmap)

    # Show the plot
    plt.show()
