from data.emojis import EMOJI_TUPLES, EMOJI_SIZE
from src.vae import VariationalAutoencoder
from src.functions import *
import plotly.graph_objects as go
from plots.various_plots import plot_error
from plotly.subplots import make_subplots
from latent_space import plot_latent_space, plot_transformation
from src.optimization import MomentumOptimizer, AdamOptimizer

LEARNING_CONSTANT = 10 ** -3  # -> *
BETA = 0.3  # -> Todo estos parametros van en la creacion del optimizador
SAVE_WEIGHTS = True
ERROR_PLOT_TITLE = "Error by Steps. Adam"



def add_heatmap_trace(fig, original, created, colorscale):
    input_letter = np.reshape(np.array(original), EMOJI_SIZE)
    output_letter = np.reshape(created, EMOJI_SIZE)
    fig.add_trace(go.Heatmap(z=np.flipud(input_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 2, col=1 + 2 * (i % 2))
    fig.add_trace(go.Heatmap(z=np.flipud(output_letter),
                             coloraxis="coloraxis"),
                  row=1 + i // 2, col=2 + 2 * (i % 2))


if __name__ == '__main__':
    data = [(font, font) for font in EMOJI_TUPLES]
    _encoder_layers = [120,60,20,10]
    _latent_space_dim = 2
    _decoder_layers = [10,20,60,120]

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
        # optimization=MomentumOptimizer(amount_of_layers, LEARNING_CONSTANT, BETA)
        optimizer_encoder=AdamOptimizer(amount_of_layers_encoder, alpha=0.00005),
        optimizer_decoder=AdamOptimizer(amount_of_layers_decoder, alpha=0.00005)
    )
    autoencoder.train(50000, 0.01, _print=True)
    # autoencoder.load_weights("./weights/weights.json")
    fig = make_subplots(rows=6, cols=4)
    colorscale = [[0, 'white'], [1, 'black']]
    for i, _font in enumerate(EMOJI_TUPLES):
        result = autoencoder.run_input(_font)[0]
        add_heatmap_trace(fig, _font, (result + 1) / 2, colorscale)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout(coloraxis=dict(colorscale=colorscale))
    fig.update_layout(title_text=f"Autoencoder Results<br><sup>Epochs: {autoencoder.i} - "
                                 f"Layers: {_encoder_layers + [_latent_space_dim] + _decoder_layers}</sup>")
    fig.show()

    plot_error(autoencoder.steps, autoencoder.errors, ERROR_PLOT_TITLE)

    if _latent_space_dim == 2:
        plot_latent_space(autoencoder, EMOJI_TUPLES,
                          ['big_smile', 'heart_eyes', 'joy', 'kiss', 'poop', 'sad', 'smile', 'sunglasses', 'surprise',
                           'sweat_smile', 'very_big_smile', 'wink'],
                          EMOJI_SIZE, False, [[0, 'white'], [1, 'black']], 20)

    plot_transformation(autoencoder, 5, EMOJI_TUPLES[0], EMOJI_TUPLES[2], False, EMOJI_SIZE)
    plot_all_transformations(autoencoder, 5, EMOJI_TUPLES, False, EMOJI_SIZE)
