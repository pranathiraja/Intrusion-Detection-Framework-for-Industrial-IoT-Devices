from flask import Flask, render_template
from Data import generate_model_metrics, detect_cyber_attack
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def generate_graph(model1_metrics, model2_metrics):
    """
    Generate a comparison graph based on the model metrics.
    """
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    model1_values = list(model1_metrics)
    model2_values = list(model2_metrics)

    x = range(len(labels))
    plt.figure(figsize=(10, 5))
    plt.plot(x, model1_values, marker='o', label='Model 1')
    plt.plot(x, model2_values, marker='s', label='Model 2')
    plt.xticks(x, labels)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.legend()
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the buffer to base64
    graph = base64.b64encode(buf.read()).decode('utf-8')
    
    # Close the plot
    plt.close()

    return graph

@app.route('/')
def index():
    # Generate model metrics
    model1_metrics = generate_model_metrics(1)
    model2_metrics = generate_model_metrics(2)

    

    # Detect cyber attacks
    model1_attack = detect_cyber_attack()
    model2_attack = detect_cyber_attack()

    # Generate comparison graph
    graph = generate_graph(model1_metrics, model2_metrics)

    return render_template('index.html',
                           model1_accuracy=model1_metrics[0],
                           model1_precision=model1_metrics[1],
                           model1_recall=model1_metrics[2],
                           model1_f1=model1_metrics[3],
                           #model1_anomaly=model1_anomaly,
                           model1_attack=model1_attack,
                           model2_accuracy=model2_metrics[0],
                           model2_precision=model2_metrics[1],
                           model2_recall=model2_metrics[2],
                           model2_f1=model2_metrics[3],
                           #model2_anomaly=model1_anomaly,
                           model2_attack=model1_attack,
                           graph=graph)

if __name__ == '__main__':
    app.run(debug=True)
