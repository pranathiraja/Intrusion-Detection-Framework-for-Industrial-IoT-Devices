from flask import Flask, render_template
import helper

app = Flask(__name__)

@app.route('/')
def index():
    # Load the network training set
    X_train, y_train = helper.load_network_train_set()
    
    # Train the network model
    network_model = helper.train_network_model(X_train, y_train)
    
    # Get and print metrics
    helper.get_and_print_metrics(network_model, X_train, y_train)
    
    # Save the global model
    global_model_file = './received_models/global_model.joblib'
    helper.save_global_model(network_model, global_model_file)
    
    # Generate model file for each client
    helper.generate_client_models(network_model, salt=b'your_salt_here')
    
    # Create a dictionary with outputs
    outputs = {
        'global_model_file': global_model_file,
        'client_model_files': helper.get_client_model_files(),
    }
    
    return render_template('index.html', outputs=outputs)

if __name__ == '__main__':
    app.run(debug=True)
