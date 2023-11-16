from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from latent_space import plot_latent_space
from src.optimization import MomentumOptimizer, AdamOptimizer

LEARNING_CONSTANT = 10 ** -3  # -> *
BETA = 0.3  # -> Todo estos parametros van en la creacion del optimizador
SAVE_WEIGHTS = True


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
    data = [(font, font) for font in FONTS_BIT_TUPLES]
    _encoder_layers = [64, 64, 64, 64, 64]
    _latent_space_dim = 2
    _decoder_layers = [64, 64, 64, 64, 64]

    amount_of_layers = len(_encoder_layers) + 1 + len(_decoder_layers) + 1
    autoencoder = Autoencoder(
        encoder_layers=_encoder_layers,
        latent_space_dim=_latent_space_dim,
        decoder_layers=_decoder_layers,
        data=data,
        activation_function=hiperbolic,
        derivation_function=hiperbolic_derivative,
        normalization_function=hiperbolic_normalization,
        # optimization=MomentumOptimizer(amount_of_layers, LEARNING_CONSTANT, BETA)
        optimization=AdamOptimizer(amount_of_layers)
    )
    autoencoder.train(10000, 0.0001, _print=True)
    # autoencoder.load_weights("./weights/weights.json")
    fig = make_subplots(rows=8, cols=8)
    colorscale = [[0, 'white'], [1, 'black']]
    for i, _font in enumerate(FONTS_BIT_TUPLES):
        result = autoencoder.run_input(_font)
        add_heatmap_trace(fig, _font, (round(result[0]) + 1) / 2, colorscale)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(coloraxis=dict(colorscale=colorscale))
    fig.update_layout(title_text=f"Autoencoder Results<br><sup>Epochs: {autoencoder.i} - "
                                 f"Layers: {_encoder_layers + [_latent_space_dim] + _decoder_layers}</sup>")
    fig.show()

    plot_latent_space(autoencoder, FONTS_BIT_TUPLES)

    if SAVE_WEIGHTS:
        autoencoder.save_weights("./weights.json")
