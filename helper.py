import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def get_host_port():
    host = '127.0.0.1'
    port = 8091
    return host, port

def split_dataset(test_size=0.2):
    df = pd.read_csv('./datasets/merged_data.csv')
    train_set, test_set = train_test_split(df, test_size=test_size, shuffle=True)
    return train_set, test_set

def split_train_set():
    train_set, _ = split_dataset()
    sensor_train = train_set.iloc[:, list(range(3)) + [-1]]
    network_train = train_set.iloc[:, 3:17]
    return sensor_train, network_train

def load_test_set():
    _, test_set = split_dataset()
    sensor_test = test_set.iloc[:, list(range(3)) + [-1]]
    network_test = test_set.iloc[:, 3:17]
    return sensor_test, network_test

def load_sensor_train_set(client_id: int):
    if 0 <= client_id <= 2:
        sensor_train, _ = split_train_set()
        x = sensor_train.iloc[:, :-1]
        y = sensor_train.iloc[:, -1]

        random_choose = np.random.choice(x.index, (len(x) % 3), replace=False)
        x = x.drop(random_choose)
        y = y.drop(random_choose)

        x_train, y_train = np.split(x, 3), np.split(y, 3)
        return x_train[client_id], y_train[client_id]
    else:
        print("Error: The client number exceeds the default of 3. Please modify the code to accept more clients.")
        return

def load_network_train_set():
    _, network_train = split_train_set()
    x_train = network_train.iloc[:, :-1]
    y_train = network_train.iloc[:, -1]
    return x_train, y_train

def get_label_ratio(y_train) -> float:
    ratio = round(np.sum(y_train == 0) / len(y_train), 8)
    return ratio

def print_label_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    label_counts = dict(zip(unique, counts))
    print("Target distribution:", label_counts, '\n')

def get_metrics(y_test, y_pred, printout=False):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    if printout:
        line = "-" * 29
        print(line)
        print(f"Accuracy : {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall   : {recall}")
        print(f"F1 Score : {f1}")
        print(line + '\n\n')

    return accuracy, precision, recall, f1

def generate_key(ratio, salt):
    seed = str(ratio).encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(seed)
    return base64.urlsafe_b64encode(key)

def train_network_model(X_train, y_train):
    network_model = RandomForestClassifier()
    network_model.fit(X_train, y_train)
    return network_model

def get_and_print_metrics(model, X, y):
    y_pred = model.predict(X)
    accuracy, precision, recall, f1 = get_metrics(y, y_pred, printout=True)

def save_global_model(model, filename):
    joblib.dump(model, filename=filename)

def generate_client_models(global_model, salt, num_clients=3):
    client_models = []

    for client_id in range(num_clients):
        # Placeholder logic, replace with actual client model training
        x_train_client, y_train_client = load_sensor_train_set(client_id)
        client_model = train_network_model(x_train_client, y_train_client)

        # Save the client model
        client_model_filename = f"client_model_{client_id}.joblib"
        joblib.dump(client_model, filename=client_model_filename)
        client_models.append(client_model_filename)

    return client_models

def get_client_model_files():
    # Assuming client models are stored in the same directory with a specific naming convention
    model_files = [file for file in os.listdir() if file.startswith("client_model_")]
    return model_files

# Example usage:
# Uncomment the lines below to test the functions

# sensor_train, network_train = split_train_set()
# sensor_test, network_test = load_test_set()
# x_train, y_train = load_sensor_train_set(0)
# x_train_net, y_train_net = load_network_train_set()

# global_model = train_network_model(x_train_net, y_train_net)
# get_and_print_metrics(global_model, x_train_net, y_train_net)
# save_global_model(global_model, "global_model.joblib")

# client_models = generate_client_models(global_model, salt)
# for client_model_file in client_models:
#     print(f"Client Model File: {client_model_file}")

# client_model_files = get_client_model_files()
# for client_model_file in client_model_files:
#     print(f"Client Model File: {client_model_file}")
