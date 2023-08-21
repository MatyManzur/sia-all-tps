import json
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():
    with(open('../multiple_test_results.json', 'r') as f):
        results = json.load(f)
        data = {}
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Execution time", "Cost", "Expanded nodes", "Frontier nodes"))
    for result in results:
        algorithm = result['algorithm']['algorithm']
        if algorithm == 'A*' or algorithm == 'GlobalGreedy' or algorithm == 'LocalGreedy':
            heuristic = result['algorithm']['heuristic']
            heuristic = heuristic.split('_heuristic')[0]
            algorithm = algorithm + ' ' + heuristic
        if algorithm == 'IDDFS':
            depth_increment = result['algorithm']['depth_increment']
            algorithm = algorithm + ' ' + str(depth_increment)
        data[algorithm] = {}
        data[algorithm]['cost'] = result['cost']
        data[algorithm]['expanded_nodes'] = result['expanded_nodes']
        data[algorithm]['frontier_nodes'] = result['frontier_nodes']
        data[algorithm]['execution_time'] = result['execution_time']
    df = pd.DataFrame.from_dict(data, orient='index')

    fig.add_trace(
        px.bar(df, x=df.index, y='execution_time', title='Time by algorithm',
               labels={'index': 'Algorithm', 'execution_time': 'Time'}).data[0],
        row=1,
        col=1
    )
    fig.add_trace(
        px.bar(df, x=df.index, y='cost', text_auto=True, title='Cost by algorithm',
               labels={'index': 'Algorithm', 'cost': 'Cost'}).data[0],
        row=1,
        col=2
    )
    fig.add_trace(
        px.bar(df, x=df.index, y='expanded_nodes', title='Expanded Nodes by algorithm',
               labels={'index': 'Algorithm', 'expanded_nodes': 'Expanded Nodes'}).data[0],
        row=2,
        col=1
    )
    fig.add_trace(
        px.bar(df, x=df.index, y='frontier_nodes', title='Frontier Nodes by algorithm',
               labels={'index': 'Algorithm', 'frontier_nodes': 'Frontier Nodes'}).data[0],
        row=2,
        col=2
    )
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_xaxes(title_text="Algorithms", row=1, col=1)
    fig.update_xaxes(title_text="Algorithms", row=1, col=2)
    fig.update_xaxes(title_text="Algorithms", row=2, col=2)
    fig.update_xaxes(title_text="Algorithms", row=2, col=1)

    fig.update_yaxes(title_text="Execution time [s]", row=1, col=1)
    fig.update_yaxes(title_text="Cost", row=1, col=2)
    fig.update_yaxes(title_text="Frontier nodes", row=2, col=2)
    fig.update_yaxes(title_text="Expanded nodes", row=2, col=1)

    fig.show()


if __name__ == '__main__':
    main()
