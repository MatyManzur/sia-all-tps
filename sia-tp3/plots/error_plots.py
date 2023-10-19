import json
import plotly.graph_objects as go

data = json.load(open("./results/test-sound-momentum6-0.json", mode='r'))

# Extract data for plotting
epochs = [entry['epoch'] for entry in data['iterations']]
error = [entry['error'] for entry in data['iterations']]
test_error = [entry['test_error'] for entry in data['iterations']]

# Create a Plotly figure with both error and test_error
fig = go.Figure()

fig.add_trace(go.Scatter(x=epochs, y=error, mode='lines', name='Training Error'))
fig.add_trace(go.Scatter(x=epochs, y=test_error, mode='lines', name='Test Error'))

fig.update_layout(
    title='Error and Test Error by Epoch',
    xaxis=dict(title='Epoch'),
    yaxis=dict(title='Error/Test Error')
)

# Display the combined plot
fig.show()
