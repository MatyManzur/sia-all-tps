import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def accuracy_test_train():
    data = json.load(open("../results/test-even-test-train-metrics-global.json", "r"))

    numbers = [int(key) for key in data.keys()]
    train_accuracy = [data[key]["train_accuracy"] for key in data.keys()]
    test_accuracy = [data[key]["test_accuracy"] for key in data.keys()]


    fig = go.Figure()


    fig.add_trace(go.Scatter(x=numbers, y=train_accuracy, mode='lines+markers', name='Train Accuracy'))
    fig.add_trace(go.Scatter(x=numbers, y=test_accuracy, mode='lines+markers', name='Test Accuracy'))


    fig.update_layout(
        title='Train and Test Accuracy',
        xaxis=dict(title='Size of Test Set'),
        yaxis=dict(title='Accuracy'),
    )

    # Show the plot
    fig.show()


if __name__ == '__main__':
    accuracy_test_train()