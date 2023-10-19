import json
import plotly.graph_objects as go
from src.metrics import accuracy, precision



def generate_data(filename):
    iterations = json.load(open(filename, mode='r'))["iterations"]
    data = iterations[-1]
    x = data["epoch"]
    y = data["error"]
    accuracy_data_test = accuracy(data["class_metrics_test"][0]) # Me quedo solo con la del 0
    precision_data_test = precision(data["class_metrics_test"][0])
    accuracy_data_train = accuracy(data["class_metrics_train"][0]) # Me quedo solo con la del 0
    precision_data_train = precision(data["class_metrics_train"][0])
    return (x, y, accuracy_data_test, precision_data_test)

iterations = 10
types = {
    "p15": [],
    "p30": [],
    "p50": [],
}

for i in range(iterations):
    types["p15"].append(generate_data(f"results/test-sound-p{15}-{i}.json"))
    types["p30"].append(generate_data(f"results/test-sound-p{30}-{i}.json"))
    types["p50"].append(generate_data(f"results/test-sound-p{50}-{i}.json"))

print(types)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y,mode='lines',name='lines'))
#fig.add_trace(go.Scatter(x=x, y=y,mode='lines+markers',name='lines+markers'))
#fig.add_trace(go.Scatter(x=x, y=y,mode='markers', name='markers'))

# Edit the layout
fig.update_layout(title='Error Por Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Error')


fig.show()

x = x[0:3000:1]
# Puse máximo 3000
accuracy_data_test = accuracy_data_test[0:3000:1]
accuracy_data_train = accuracy_data_train[0:3000:1]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=accuracy_data_test, mode='lines', name='test'))
fig.add_trace(go.Scatter(x=x, y=accuracy_data_train, mode='lines', name='train'))
fig.update_layout(title='Accuracy of Class 0 per Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Accuracy')
fig.show()

# Puse máximo 3000
precision_data_test = precision_data_test[0:3000:1]
precision_data_train = precision_data_train[0:3000:1]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=precision_data_test, mode='lines', name='test'))
fig.add_trace(go.Scatter(x=x, y=precision_data_train, mode='lines', name='train'))
fig.update_layout(title='Precision of Class 0 per Epoch',
                   xaxis_title='Epoch',
                   yaxis_title='Precision')
fig.show()