import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.ej3_digitos_ruido import DATA_DIGITOS_RUIDO_SHOW

matrices = [np.fliplr(np.array(matrix)[::-1].reshape(7,5)) for matrix, _ in DATA_DIGITOS_RUIDO_SHOW]

num_cols = len(matrices)

fig = make_subplots(rows = 1, cols = num_cols, subplot_titles=[f'' for i in range(len(matrices))])

traces = []
for i, matrix in enumerate(matrices):
    heatmap = go.Heatmap(z=matrix, colorscale=[[0, 'white'], [1, 'black']])
    row = 1
    cols = (i % num_cols) + 1
    fig.add_trace(heatmap, row = row, col = cols)

fig.update_layout(
    title="Number Heatmap",
    xaxis=dict(title="X-axis"),
    yaxis=dict(title="Y-axis")
)

fig.show()
