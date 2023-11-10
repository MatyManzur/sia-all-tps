from data.fonts import FONTS_BIT_TUPLES
from src.autoencoder import Autoencoder
from src.functions import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == '__main__':
    data = [(font, font) for font in FONTS_BIT_TUPLES]
    autoencoder = Autoencoder(
        encoder_layers=[15],
        latent_space_dim=2,
        decoder_layers=[15],
        data=data,
        activation_function=sigmoid,
        derivation_function=sigmoid_derivative,
        normalization_function=sigmoid_normalization,
        optimization={
            "type": "momentum",
            "beta": 0.3
        }
    )
    autoencoder.train(10000, 0.3)
    fig = make_subplots(rows=8, cols=8)
    colorscale = [[0, 'white'], [1, 'black']]
    for i, _font in enumerate(FONTS_BIT_TUPLES):
        result = autoencoder.run_input(_font)
        input_letter = np.reshape(np.array(_font), [7, 5])
        output_letter = np.reshape(result[0], [7, 5])
        fig.add_trace(go.Heatmap(z=np.flipud(input_letter), colorscale=colorscale), row=1 + i // 4, col=1 + 2*(i % 4))
        fig.add_trace(go.Heatmap(z=np.flipud(output_letter), colorscale=colorscale), row=1 + i // 4, col=2 + 2*(i % 4))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_coloraxes(showscale=False)
    fig.show()
