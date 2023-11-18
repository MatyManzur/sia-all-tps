from data.emojis import EMOJI_TUPLES
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
    input_letter = np.reshape(np.array(original), [24, 24])
    output_letter = np.reshape(created, [24, 24])
    fig.add_trace(go.Heatmap(z=np.flipud(input_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 2, col=1 + 2 * (i % 2))
    fig.add_trace(go.Heatmap(z=np.flipud(output_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 2, col=2 + 2 * (i % 2))


if __name__ == '__main__':
    data = [(font, font) for font in EMOJI_TUPLES]
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
        optimization=AdamOptimizer(amount_of_layers)
    )
    autoencoder.train(1000, 0.8, _print=True)
    fig = make_subplots(rows=6, cols=4)
    colorscale = [[0, 'white'], [1, 'black']]
    for i, _font in enumerate(EMOJI_TUPLES):
        result = autoencoder.run_input(_font)
        add_heatmap_trace(fig, _font, (result[0]+1)/2, colorscale)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(coloraxis=dict(colorscale=colorscale))
    fig.update_layout(title_text=f"Autoencoder Results<br><sup>Epochs: {autoencoder.i} - "
                                 f"Layers: {_encoder_layers + [_latent_space_dim] + _decoder_layers}</sup>")
    fig.show()

    plot_latent_space(autoencoder, EMOJI_TUPLES,
                      ['big_smile', 'heart_eyes', 'joy', 'kiss', 'poop', 'sad', 'smile', 'sunglasses', 'surprise', 'sweat_smile', 'very_big_smile', 'wink'],
                      'poop', [24,24], False)

