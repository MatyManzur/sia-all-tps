import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json

def plot_error(steps, errors, plot_title): 
    fig = go.Figure(data=go.Scatter(x=steps, y=errors))
    
    fig.update_layout(title=plot_title,
                   xaxis_title='Steps',
                   yaxis_title='Min Error')

    fig.show()

def plot_multiple_errors(errors:dict):
    data = []
    for test in errors['tests']:
        aux = []
        for iteration in test['data']:
            aux.append(iteration['errors'])
        aux = np.array(aux)
        mean = np.mean(aux, axis=0)
        std = np.std(aux, axis=0)
        label = str(test['layers'])
        for i in range(len(mean)):
            data.append([i, mean[i], std[i], label])
    df = pd.DataFrame(data, columns=['Epoch', 'ECM', 'std', 'layers'])
    print(df)
    fig = px.line(df, x="Epoch", y="ECM", color="layers", title="Error(ECM) by epoch using hyperbolic "
                                                                                "tangent and ADAM optimizer", log_y=True)
    fig.show()


if __name__ == '__main__':
    errors = json.load(open("results_autoencoder_neurons.json", "r"))
    plot_multiple_errors(errors)