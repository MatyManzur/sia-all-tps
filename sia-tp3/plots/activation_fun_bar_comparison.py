import json

import numpy as np
import plotly.express as px


def xor():

    # Your dictionary data
    data = {
        'hiperbolic': {
            'values': [],
        },
        'logistic': {
            'values': [],
        },
        'sign': {
            'values': [],
        }
    }

    for i in range(20):
        raw_data = json.load(open(f"../results/xor/test-hiperbolic-{i}.json", mode='r'))
        data['hiperbolic']['values'].append(raw_data['epoch_reached'])
        raw_data = json.load(open(f"../results/xor/test-logistic-{i}.json", mode='r'))
        data['logistic']['values'].append(raw_data['epoch_reached'])
        raw_data = json.load(open(f"../results/xor/test-sign-{i}.json", mode='r'))
        data['sign']['values'].append(raw_data['epoch_reached'])
    data['hiperbolic']['average'] = np.mean(data['hiperbolic']['values'])
    data['hiperbolic']['error'] = np.std(data['hiperbolic']['values'])
    data['logistic']['average'] = np.mean(data['logistic']['values'])
    data['logistic']['error'] = np.std(data['logistic']['values'])
    data['sign']['average'] = np.mean(data['sign']['values'])
    data['sign']['error'] = np.std(data['sign']['values'])



    # Extract the labels and values
    labels = list(data.keys())
    averages = [item['average'] for item in data.values()]
    errors = [item['error'] for item in data.values()]

    # Create a bar plot
    fig = px.bar(x=labels, y=averages, error_y=errors, title="Necessary epochs for reaching an error of 0.01")
    fig.update_layout(xaxis_title='Activation Functions', yaxis_title='Epochs')
    fig.show()

def odd_even():

    # Your dictionary data
    data = {
        'hiperbolic': {
            'values': [],
        },
        'logistic': {
            'values': [],
        }
    }

    for i in range(20):
        raw_data = json.load(open(f"../results/odd_even/test-hiperbolic-{i}.json", mode='r'))
        data['hiperbolic']['values'].append(raw_data['epoch_reached'])
        raw_data = json.load(open(f"../results/odd_even/test-logistic-{i}.json", mode='r'))
        data['logistic']['values'].append(raw_data['epoch_reached'])
    data['hiperbolic']['average'] = np.mean(data['hiperbolic']['values'])
    data['hiperbolic']['error'] = np.std(data['hiperbolic']['values'])
    data['logistic']['average'] = np.mean(data['logistic']['values'])
    data['logistic']['error'] = np.std(data['logistic']['values'])



    # Extract the labels and values
    labels = list(data.keys())
    averages = [item['average'] for item in data.values()]
    errors = [item['error'] for item in data.values()]

    # Create a bar plot
    fig = px.bar(x=labels, y=averages, error_y=errors, title="Necessary epochs for reaching an error of 0.01")
    fig.update_layout(xaxis_title='Activation Functions', yaxis_title='Epochs')
    fig.show()

if __name__ == '__main__':
    xor()