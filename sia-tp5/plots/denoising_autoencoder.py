from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
from plots.various_plots import plot_error
from plotly.subplots import make_subplots
from latent_space import plot_latent_space
from src.optimization import MomentumOptimizer, AdamOptimizer

LEARNING_CONSTANT = 10 ** -3  # -> *
BETA = 0.3  # -> Todo estos parametros van en la creacion del optimizador
SAVE_WEIGHTS = True
ERROR_PLOT_TITLE = "Error by Steps. Adam"


def add_heatmap_trace(fig, original, created, colorscale, i):
    input_letter = np.reshape(np.array(original), [7, 5])
    output_letter = np.reshape(created, [7, 5])
    fig.add_trace(go.Heatmap(z=np.flipud(input_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 4, col=1 + 2 * (i % 4))
    fig.add_trace(go.Heatmap(z=np.flipud(output_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 4, col=2 + 2 * (i % 4))

def gaussian_noise(tuple, mean, std):
    new_tuple = []
    for bit in tuple:
        new_tuple.append(bit + np.random.normal(mean, std))
    return new_tuple

def salt_and_pepper(tuple, prob):
    new_tuple = []
    for bit in tuple:
        if np.random.random() < prob:
            new_tuple.append(1 - bit)
        else:
            new_tuple.append(bit)
    return new_tuple

def poisson(figure, figure_size, mean):
    noisemap = np.ones((figure_size)) * mean #mean
    noisy = figure + np.random.poisson(noisemap)
    return noisy

S_P_NOISE = 0.04

if __name__ == '__main__':
    data = [(font, font) for font in FONTS_BIT_TUPLES]
    fig = make_subplots(rows=8, cols=8)
    noise_func = lambda font: salt_and_pepper(font, S_P_NOISE)
    # noise_func = lambda font: gaussian_noise(font, 0, 0.1)
    # noise_func = lambda font: poisson(font, 35, 0.1)

    noisy_data = [(noise_func(font), font) for font in FONTS_BIT_TUPLES]
    _encoder_layers = [64, 64, 64, 64, 64]
    _latent_space_dim = 2
    _decoder_layers = [64, 64, 64, 64, 64]
    amount_of_layers = len(_encoder_layers) + 1 + len(_decoder_layers) + 1

    autoencoder = Autoencoder(
        encoder_layers=_encoder_layers,
        latent_space_dim=_latent_space_dim,
        decoder_layers=_decoder_layers,
        data=noisy_data,
        activation_function=hiperbolic,
        derivation_function=hiperbolic_derivative,
        normalization_function=hiperbolic_normalization,
        # optimization=MomentumOptimizer(amount_of_layers, LEARNING_CONSTANT, BETA)
        optimization=AdamOptimizer(amount_of_layers)
    )

    autoencoder.train(2000, 0.0001)
    extra_noisy_data = [(noise_func(font), font) for font in FONTS_BIT_TUPLES]
    colorscale = [[0, 'white'], [1, 'black']]

    for i, (noisy, data) in enumerate(extra_noisy_data):
        result = autoencoder.run_input(noisy)
        add_heatmap_trace(fig, noisy, (round(result[0]) + 1) / 2, colorscale, i)
        
    fig.show()


