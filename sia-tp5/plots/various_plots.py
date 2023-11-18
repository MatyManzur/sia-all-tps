import plotly.graph_objects as go
import numpy as np


def plot_error(steps, errors, plot_title): 
    fig = go.Figure(data=go.Scatter(x=steps, y=errors))
    
    fig.update_layout(title=plot_title,
                   xaxis_title='Steps',
                   yaxis_title='Min Error')

    fig.show()
    