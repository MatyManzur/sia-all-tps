import json
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def functions_plot():
    results = json.load(open("results/ej_2_function_test.json", "r"))
    sub = make_subplots(rows=1, cols=2,
                        subplot_titles=("Sigmoid Activation Function", "Hyperbolic Activation Function"))
    output = results[0]['output']['iterations'][0:500]
    df = pd.DataFrame(output, columns=['Epoch', 'Training', 'Test'])
    melt = df.melt(id_vars=['Epoch'], value_vars=['Training', 'Test'], var_name='Data Set', value_name='Error')
    line = px.line(melt, x='Epoch', y='Error', color='Data Set',title='Error by Epoch with Sigmoid Activation '
                                                                      'Function with Half Testing and Training Data')
    line.show()


def functions_linear_plot():
    results = json.load(open("results/ej_2_function_test.json", "r"))
    sub = make_subplots(rows=1, cols=2,
                        subplot_titles=("Sigmoid Activation Function", "Hyperbolic Activation Function"))
    output = results[2]['output']['iterations'][0:500]
    df = pd.DataFrame(output, columns=['Epoch', 'Training', 'Test'])
    # melt = df.melt(id_vars=['Epoch'], value_vars=['Training', 'Test'], var_name='Data Set', value_name='Error')
    line = px.line(df, x='Epoch', y='Training',
                   title='Error by Epoch with Linear Activation - Full Data')
    line.update_layout(yaxis_title='Error', yaxis_type="log")
    line.show()

def learning_plot():
    results = json.load(open("results/ej_2_learning_test.json", "r"))
    data = []
    for result in results:
        constant = result['learning_constant']
        output = result['output']
        for d in output[:2500]:
            data.append([d[0],d[1],constant])
    df = pd.DataFrame(data, columns=['Epoch', 'Error', 'Learning Constant'])
    print(df)
    line = px.line(df, x='Epoch', y='Error', color='Learning Constant',
                   title='Error by Epoch and Learning Constant with Identity Activation Function with Half Testing and Training Data')
    line.show()


def data_parting_plot():
    results = json.load(open("results/ej_2_data_parting_test.json", "r"))
    data_sigmoid = []
    data_hyperbolic = []
    for result in results:
        function = result['function']
        min_error = result['min_error']
        min_error_testing = result['min_error_testing']
        percentage = len(result['data'][0]) * 100 / 28

        if function == 'sigmoid':
            data_sigmoid.append([percentage, min_error, min_error_testing])
        else:
            data_hyperbolic.append([percentage, min_error, min_error_testing])

    df_sigmoid = pd.DataFrame(data_sigmoid, columns=['Percentage', 'Error Learning', 'Error Testing'])
    df_sigmoid = df_sigmoid.sort_values(by=['Percentage'])
    df_hyperbolic = pd.DataFrame(data_hyperbolic, columns=['Percentage', 'Error Learning', 'Error Testing'])
    df_hyperbolic = df_hyperbolic.sort_values(by=['Percentage'])
    px.line(df_sigmoid, x='Percentage', y= ['Error Learning', 'Error Testing'],
               title='Error Based on Percentage of Data Used for Learning - Sigmoid', markers='lines+markers').show()
    px.line(df_hyperbolic, x='Percentage', y=['Error Learning', 'Error Testing'],
               title='Error Based on Percentage of Data - Hyperbolic', markers='lines+markers').show()
    # print(df_sigmoid)
    # print()
    # print(df_hyperbolic)

if __name__ == '__main__':
    # functions_plot()
    # learning_plot()
    data_parting_plot()