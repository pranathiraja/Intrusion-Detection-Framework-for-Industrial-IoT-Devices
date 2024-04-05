import helper
import socket
import joblib
from sklearn.ensemble import RandomForestClassifier
import os
import time
from cryptography.fernet import Fernet
import zlib
import warnings
warnings.simplefilter('ignore')


client_id = 2
print(f"Client {client_id}:\n")

# Get the dataset for local model
X_train, y_train = helper.load_sensor_train_set(client_id - 1)
helper.print_label_distribution(y_train)
ratio = helper.get_label_ratio(y_train)

# Create and train the local model
model = RandomForestClassifier()
train_start = time.time()
model.fit(X_train, y_train)
train_end = time.time()
train_time = train_end - train_start
print(f"Client {client_id} model training completed in {train_time:.4f} seconds.")

# Generate model file
path = "./client_models"
os.makedirs(path, exist_ok=True)
filename = f'{path}/client_{client_id}.joblib'
joblib.dump(model, filename=filename)
print(f"Client {client_id} model saved to: {filename}")

# Generate and display encryption information
salt = os.urandom(16)
key = helper.generate_key(ratio, salt)
print(f"\nClient {client_id} model encryption information:")
print(f"Key: {key}\nSeed: {ratio}\nSalt: {salt}\n")

# Read model file
with open(filename, 'rb') as file:
    model_file = file.read()
print(f"Model file size: {(len(model_file) / (1024 ** 2)):.4f} MB")

# Encrypt the model file
encr_start = time.time()
encrypted_model = Fernet(key).encrypt(model_file)
encr_end = time.time()
print(f"Model file encrypted, file size: {(len(encrypted_model) / (1024 ** 2)):.4f} MB, "
      f"time spent: {encr_end - encr_start:.4f} seconds")

# Compress the model file
comp_start = time.time()
compressed_model = zlib.compress(encrypted_model)
comp_end = time.time()
print(f"Model file compressed, file size: {(len(compressed_model) / (1024 ** 2)):.4f} MB, "
      f"time spent: {comp_end - comp_start:.4f} seconds")

# Define host and port
host, port = helper.get_host_port()

try:
    # Connect to the server
    client = socket.socket()
    client.connect((host, port))
    print(f"\nClient {client_id} has connected to the server")

    # Send salt, client id, file size, ratio
    client.sendall(salt)

    file_size = len(compressed_model)
    metadata = f'{client_id},{file_size},{ratio}\n'.encode('utf-8')
    client.sendall(metadata)

    # Send encrypted and compressed model
    start_time = time.time()
    client.sendall(compressed_model)
    end_time = time.time()

    print(f"Model file has been sent from Client {client_id}, "
          f"file size: {(file_size / (1024 ** 2)):.4f} MB, "
          f"time spent: {end_time - start_time:.4f} seconds")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()

print("-" * 35 + f" Client {client_id} Completed " + "-" * 35)
