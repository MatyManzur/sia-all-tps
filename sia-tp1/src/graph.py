import json
import pandas as pd
import plotly.express as px


def main():
    with(open('../multiple_test_results.json', 'r') as f):
        results = json.load(f)
        data = {}
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
    px.bar(df,x=df.index,y='execution_time',title='Time by algorithm',labels={'index':'Algorithm','execution_time':'Time'}).show()




if __name__ == '__main__':
    main()
