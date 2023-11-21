import json

from data.fonts import FONTS_BIT_TUPLES
from src.vae import VariationalAutoencoder
from src.functions import *
import plotly.graph_objects as go
from plots.various_plots import plot_error
from plotly.subplots import make_subplots
from latent_space import plot_latent_space
from src.optimization import AdamOptimizer

ERROR_PLOT_TITLE = "Error by Steps. Adam"


def add_heatmap_trace(fig, original, created, colorscale):
    input_letter = np.reshape(np.array(original), [7, 5])
    output_letter = np.reshape(created, [7, 5])
    fig.add_trace(go.Heatmap(z=np.flipud(input_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 4, col=1 + 2 * (i % 4))
    fig.add_trace(go.Heatmap(z=np.flipud(output_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 4, col=2 + 2 * (i % 4))


if __name__ == '__main__':
    with open("../configs/vae_config.json", "r") as f:
        config = json.load(f)

        data = [(font, font) for font in FONTS_BIT_TUPLES]
        _encoder_layers = config['encoder_layers']
        _latent_space_dim = config['latent_space_dim']
        _decoder_layers = config['decoder_layers']

        amount_of_layers_encoder = len(_encoder_layers) + 1
        amount_of_layers_decoder = len(_decoder_layers) + 1
        autoencoder = VariationalAutoencoder(
            encoder_layers=_encoder_layers,
            latent_space_dim=_latent_space_dim,
            decoder_layers=_decoder_layers,
            data=data,
            activation_function=hiperbolic,
            derivation_function=hiperbolic_derivative,
            normalization_function=hiperbolic_normalization,
            optimizer_encoder=AdamOptimizer(amount_of_layers_encoder, alpha=config['adam_alpha']),
            optimizer_decoder=AdamOptimizer(amount_of_layers_decoder, alpha=config['adam_alpha'])
        )
        autoencoder.train(config['max_epochs'], config['min_error_threshold'], _print=True)
        if config['load_weights']:
            autoencoder.load_weights(config['load_weights_file'])
        fig = make_subplots(rows=8, cols=8)
        colorscale = [[0, 'white'], [1, 'black']]
        for i, _font in enumerate(FONTS_BIT_TUPLES):
            result = autoencoder.run_input(_font)[0]
            add_heatmap_trace(fig, _font, (round(result) + 1) / 2, colorscale)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(coloraxis=dict(colorscale=colorscale))
        fig.update_layout(title_text=f"Autoencoder Results<br><sup>Epochs: {autoencoder.i} - "
                                     f"Layers: {_encoder_layers + [_latent_space_dim] + _decoder_layers}</sup>")
        fig.show()

        plot_error(autoencoder.steps, autoencoder.errors, ERROR_PLOT_TITLE)

        plot_latent_space(autoencoder, FONTS_BIT_TUPLES, should_round=False)

        if config['save_weights']:
            autoencoder.save_weights("./weights.json")
