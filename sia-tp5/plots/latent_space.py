import math

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


def plot_latent_space(autoencoder: Autoencoder | VariationalAutoencoder, font, font_names=LETTERS, font_shape=[7, 5],
                      should_round=True, colorscale=None, grid_size=20):
    if colorscale is None:
        colorscale = [[0, 'black'], [1, 'white']]
    x_coord = []
    y_coord = []
    letters = []
    min_x, max_x, min_y, max_y = (0, 0, 0, 0)

    for i, letter in enumerate(font):
        latent_space_xy = autoencoder.run_input(letter)[1]
        x = latent_space_xy[0][0]
        y = latent_space_xy[1][0]
        min_x = x if x < min_x else min_x
        max_x = x if x > max_x else max_x
        min_y = y if y < min_y else min_y
        max_y = y if y > max_y else max_y

        x_coord.append(latent_space_xy[0][0])
        y_coord.append(latent_space_xy[1][0])
        letters.append(font_names[i])
    rows = []
    step_y = (max_y - min_y)/grid_size
    step_x = (max_x - min_x)/grid_size
    for y in np.arange(min_y, max_y, step_y):
        row = []
        for x in np.arange(min_x, max_x, step_x):
            letter_at_xy = autoencoder.output_from_latent_space((x, y))
            if should_round:
                letter_at_xy = round(letter_at_xy)
            letter_at_xy = np.flipud(letter_at_xy.reshape(font_shape))
            letter_at_xy = np.insert(letter_at_xy, 0, -1, 0)
            letter_at_xy = np.insert(letter_at_xy, 0, -1, 1)
            row.append(letter_at_xy)
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)

    fig2 = go.Figure(go.Heatmap(
        z=grid,
        colorscale=colorscale,
    ))

    fig = px.scatter(x=x_coord, y=y_coord, text=letters)
    fig.update_traces(textposition='top center')

    fig.update_layout(height=800, title_text='Latent Space')

    fig.show()
    fig2.show()

    plot_interactive_latent_space(autoencoder, x_coord, y_coord, letters, should_round, font_shape)


def plot_interactive_latent_space(autoencoder, x, y, letters, should_round, font_shape):
    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.scatter(x, y)
    for index in range(len(x)):
        plt.text(x[index], y[index] * (1 + 0.03), letters[index], fontsize=12)

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


def plot_transformation(autoencoder, steps, origin, destination, should_round, font_shape=None):
    colorscale = [[0, 'white'], [1, 'black']]
    if font_shape is None:
        font_shape = [7, 5]

    fig = make_subplots(rows=math.ceil(steps / 3), cols=3)
    origin_latent_space = autoencoder.run_input(origin)[1]
    destination_latent_space = autoencoder.run_input(destination)[1]

    for i in range(steps+1):
        latent_space = origin_latent_space + (destination_latent_space - origin_latent_space) * i / steps
        letter_at_step = autoencoder.output_from_latent_space(latent_space)
        if should_round:
            letter_at_step = round(letter_at_step)
        letter_at_step = np.flipud(letter_at_step.reshape(font_shape))
        letter_at_step = np.insert(letter_at_step, 0, -1, 0)
        letter_at_step = np.insert(letter_at_step, 0, -1, 1)
        fig.add_trace(go.Heatmap(z=letter_at_step, colorscale=colorscale, coloraxis="coloraxis"),
                      row=1 + i // 3, col=1 + i % 3)

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(coloraxis=dict(colorscale=colorscale))
    fig.show()
