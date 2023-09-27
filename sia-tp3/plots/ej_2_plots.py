import json
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

def learning_plot():
    results = json.load(open("./results/ej_2_learning_test.json", "r"))
    data = []
    for result in results:
        constant = result['learning_constant']
        output = result['output']
        for d in output[:2500]:
            data.append([d[0],d[1],constant])
    df = pd.DataFrame(data, columns=['Epoch', 'Error', 'Learning Constant'])
    print(df)
    line = px.line(df, x='Epoch', y='Error', color='Learning Constant',
                   title='Error by Epoch and Learning Constant with Hyperbolic Activation Function with Half Testing and Training Data')
    line.show()
    #

if __name__ == '__main__':
    # functions_plot()
    learning_plot()