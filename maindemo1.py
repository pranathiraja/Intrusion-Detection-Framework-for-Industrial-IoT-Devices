import helper
import aggregated_predict as ap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def smo_algorithm(X_train, y_train, X_test):
    # Implement SMO algorithm here 
    smo_model = YourSMOImplementation()
    smo_model.fit(X_train, y_train)
    smo_predictions = smo_model.predict(X_test)
    smo_predictions = None  # Placeholder for SMO predictions
    return smo_predictions

def sdpn_algorithm(X_train, y_train, X_test):
    # Implement SDPN algorithm here 
    sdpn_model = YourSDPNImplementation()
    sdpn_model.fit(X_train, y_train)
    dpn_predictions = sdpn_model.predict(X_test)
    sdpn_predictions = None  # Placeholder for SDPN predictions
    return sdpn_predictions

# Load test data
sensor_test, network_test = helper.load_test_set()

# Load models
sensor_models = ['./received_models/client_1.joblib', './received_models/client_2.joblib', './received_models/client_3.joblib']
network_model = './received_models/global_model.joblib'

# Apply SMO and SDPN algorithms
sensor_train, sensor_val = train_test_split(sensor_test, test_size=0.2, random_state=42)
network_train, network_val = train_test_split(network_test, test_size=0.2, random_state=42)
smo_predictions = smo_algorithm(sensor_train, sensor_train.iloc[:, -1], sensor_val)
sdpn_predictions = sdpn_algorithm(network_train, network_train.iloc[:, -1], network_val)

# Get predictions and metrics for existing models
models_predictions, models_metrics = ap.get_predictions_and_metrics(
    local_models=sensor_models + [smo_predictions],  # Include SMO predictions
    sensor_test=sensor_test,
    global_model=network_model,
    network_test=network_test,
)

y_test = network_test.iloc[:, -1]
model_weights = [0.2, 0.2, 0.2, 0.4]

score = ap.aggregate_predict_by_score(
    models_predictions=models_predictions,
    models_metrics=models_metrics,
    models_weights=model_weights
)

vote = ap.aggregate_predict_by_vote(models_predictions=models_predictions)
vote_metrics = helper.get_metrics(y_test, vote, printout=False)

metrics = {
    "score": score,
    "vote": vote_metrics
}

# Printing metrics
print("Demo 1 Output:")
print("Score Metrics:")
print("Accuracy : {:.2f}".format(metrics['score']['accuracy']))
print("Precision: {:.2f}".format(metrics['score']['precision']))
print("Recall   : {:.2f}".format(metrics['score']['recall']))
print("F1 Score : {:.2f}".format(metrics['score']['f1_score']))
print("----------------------------")
print("Vote Metrics:")
print("Accuracy : {:.2f}".format(metrics['vote']['accuracy']))
print("Precision: {:.2f}".format(metrics['vote']['precision']))
print("Recall   : {:.2f}".format(metrics['vote']['recall']))
print("F1 Score : {:.2f}".format(metrics['vote']['f1_score']))
print("----------------------------")

# Predict the type of anomaly based on thresholds (adjust as needed)
threshold_precision = 0.8
threshold_recall = 0.8
threshold_f1 = 0.8

if metrics['score']['precision'] > threshold_precision and metrics['score']['recall'] > threshold_recall and metrics['score']['f1_score'] > threshold_f1:
    confidence_level = "High"
else:
    confidence_level = "Low"

# Determine the type of anomaly based on specific conditions (replace with your own logic)
if metrics['score']['precision'] > 0.85 and metrics['score']['recall'] > 0.85:
    predicted_anomaly_type = "DoS Attack"
elif metrics['score']['precision'] > 0.8 and metrics['score']['recall'] > 0.8:
    predicted_anomaly_type = "Probe Attack"
else:
    predicted_anomaly_type = "Unknown"

print("Confidence Level:", confidence_level)
print("Predicted Anomaly Type:", predicted_anomaly_type)
