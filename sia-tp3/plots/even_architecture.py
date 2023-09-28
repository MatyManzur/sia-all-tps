import json

import numpy as np
import pandas as pd
import plotly.express as px



def even_arch():

    data = {}

    for i in range(1,4,1):
        data[i] = {}
        for j in range(2,15,2):
            epochs = []
            for k in range(20):
                raw_data = json.load(open(f"../results/odd_even/test-{j}x{i}-{k}.json", mode='r'))
                epochs.append(raw_data['epoch_reached'])
            data[i][j] = np.average(epochs)

    df = pd.DataFrame(data)

    fig = px.imshow(df, text_auto=True, aspect='auto', labels={'color':'Epochs'})
    fig.update_layout(xaxis_title='Layer count', yaxis_title='Neurons per layer', title='Necessary epochs for reaching an error of 0.01 by Network Architecture')
    fig.show()



"""
    # Extract the labels and values
    labels = list(data.keys())
    averages = [item['average'] for item in data.values()]
    errors = [item['error'] for item in data.values()]

    # Create a bar plot
    fig = px.bar(x=labels, y=averages, error_y=errors, title="Necessary epochs for reaching an error of 0.01")
    fig.update_layout(xaxis_title='Activation Functions', yaxis_title='Epochs')
    fig.show()"""

if __name__ == '__main__':
    even_arch()