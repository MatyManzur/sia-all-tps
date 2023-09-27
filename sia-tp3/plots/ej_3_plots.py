import json
import plotly.graph_objects as go

x = []
y = []

data = json.load(open("results/network-size-test/test-64-8.json", mode='r'))
for epoch_data in data["iterations"]:
    x.append(epoch_data["epoch"])
    y.append(epoch_data["error"])


fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,mode='lines',name='lines'))
#fig.add_trace(go.Scatter(x=x, y=y,mode='lines+markers',name='lines+markers'))
#fig.add_trace(go.Scatter(x=x, y=y,mode='markers', name='markers'))

# Edit the layout
fig.update_layout(title='Error Por Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Error')


fig.show()
