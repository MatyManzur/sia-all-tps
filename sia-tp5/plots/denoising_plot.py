import json
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

if __name__ == '__main__':
  with open('./plots/denoising_results.json') as file:
    data = json.load(file)
    accuracy_values = [[result["accuracy"] for result in entry["results"]] for entry in data]

    mean_accuracies = np.mean(accuracy_values, axis=1)
    std_accuracies = np.std(accuracy_values, axis=1)

    # Create a bar chart with error bars
    fig = go.Figure()

    for i, entry in enumerate(data):
        fig.add_trace(
            go.Bar(
                x=[entry["label"]],
                y=[mean_accuracies[i]],
                error_y=dict(type="data", array=[std_accuracies[i]]),
                name=entry["label"],
            )
        )

    # Update layout
    fig.update_layout(
        title="Average Accuracy with Error Bars",
        xaxis=dict(title="Gaussian Error"),
        yaxis=dict(title="Average Accuracy"),
    )

    # Show the plot
    fig.show()


# Extract accuracy values
