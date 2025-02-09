# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
from genomic_benchmarks.data_check import info
from genomic_benchmarks.dataset_getters.pytorch_datasets import HumanEnhancersCohn
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import BackendSampler
from functools import partial
from qiskit.providers.basic_provider import BasicProvider
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Subset  # <-- Add this import


# Streamlit App Title
st.title("Federated Quantum Machine Learning (Genomic Data)")
st.write("This app simulates federated learning with a quantum model.")

# Load Dataset (Cached for Performance)
@st.cache_data
def load_data():
    info("human_enhancers_cohn", version=0)
    test_set = HumanEnhancersCohn(split='test', version=0)
    train_set = HumanEnhancersCohn(split='train', version=0)
    data_set = train_set + test_set
    return data_set, train_set, test_set

data_set, train_set, test_set = load_data()

# Reduce Dataset Size for Testing using proper Subset
st.write(f"Original dataset size: {len(data_set)}")

# Create indices for subset
subset_indices = list(range(1000))  # First 1000 samples
data_set = Subset(data_set, subset_indices)  # <-- Proper way to subset

st.write(f"Reduced dataset size: {len(data_set)}")

# Preprocess Data (No Caching)
def preprocess_data(_data_set, word_size=125):
    st.write("Preprocessing data...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    word_combinations = defaultdict(int)
    iteration = 1

    # Step 1: Build word combinations
    for idx, (text, _) in enumerate(_data_set):
        for i in range(len(text)):
            word = text[i:i+word_size]
            if word_combinations.get(word) is None:
                word_combinations[word] = iteration
                iteration += 1
        progress_bar.progress((idx + 1) / len(_data_set))
        status_text.text(f"Processing sequence {idx + 1}/{len(_data_set)}")

    # Step 2: Encode sequences
    np_data_set = []
    for idx in range(len(_data_set)):
        sequence, label = _data_set[idx]
        sequence = sequence.strip()
        words = [sequence[i:i + word_size] for i in range(0, len(sequence), word_size)]
        int_sequence = np.array([word_combinations[word] for word in words])
        data_point = {'sequence': int_sequence, 'label': label}
        np_data_set.append(data_point)
        progress_bar.progress((idx + 1) / len(_data_set))
        status_text.text(f"Encoding sequence {idx + 1}/{len(_data_set)}")

    np.random.shuffle(np_data_set)
    st.success("Preprocessing complete!")
    return np_data_set

# Preprocess the data
np_data_set = preprocess_data(data_set)

# Split Data into Train and Test
def split_data(np_data_set, train_size=750, test_size=250):  # Reduced sizes for testing
    np_train_data = np_data_set[:train_size]
    np_test_data = np_data_set[-test_size:]
    return np_train_data, np_test_data

np_train_data, np_test_data = split_data(np_data_set)


# Define Client Class
class Client:
    def __init__(self, data):
        self.models = []
        self.primary_model = None
        self.data = data
        self.test_scores = []
        self.train_scores = []

# Split Dataset for Clients
def split_dataset(num_clients, num_epochs, samples_per_epoch):
    clients = []
    for i in range(num_clients):
        client_data = []
        for j in range(num_epochs):
            start_idx = (i * num_epochs * samples_per_epoch) + (j * samples_per_epoch)
            end_idx = (i * num_epochs * samples_per_epoch) + ((j + 1) * samples_per_epoch)
            client_data.append(np_train_data[start_idx:end_idx])
        clients.append(Client(client_data))
    return clients

# Federated Aggregation Functions
def sort_epoch_results(epoch_results):
    pairs = zip(epoch_results['weights'], epoch_results['test_scores'])
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    sorted_weights, sorted_test_scores = zip(*sorted_pairs)
    return {'weights': list(sorted_weights), 'test_scores': list(sorted_test_scores)}

def scale_test_scores(sorted_epoch_results):
    min_test_score = sorted_epoch_results['test_scores'][0]
    max_test_score = sorted_epoch_results['test_scores'][-1]
    min_weight, max_weight = [0.1, 1]
    scaled_weights = [
        min_weight + (max_weight - min_weight) * (test_score - min_test_score) / (max_test_score - min_test_score)
        for test_score in sorted_epoch_results['test_scores']
    ]
    sorted_epoch_results['fl_avg_weights'] = scaled_weights
    return sorted_epoch_results

def calculate_weighted_average(model_weights, fl_avg_weights):
    weighted_sum_weights = []
    for index in range(len(model_weights[0])):
        weighted_sum_weights.append(0)
        weighted_sum_weights[index] = sum([(weights_array[index] * avg_weight) for weights_array, avg_weight in zip(model_weights, fl_avg_weights)]) / sum(fl_avg_weights)
    return weighted_sum_weights
    
    
def weighted_average(epoch_results, global_model_weights_last_epoch = None, global_model_accuracy_last_epoch = None):
  if(global_model_weights_last_epoch != None):
    epoch_results['weights'].append(global_model_weights_last_epoch)
    epoch_results['test_scores'].append(global_model_accuracy_last_epoch)
  epoch_results = sort_epoch_results(epoch_results)
  epoch_results = scale_test_scores(epoch_results)
  print(epoch_results)
  weighted_average_weights_curr_epoch = calculate_weighted_average(epoch_results['weights'], epoch_results['fl_avg_weights'])
  return weighted_average_weights_curr_epoch



def weighted_average_best_pick(epoch_results, global_model_weights_last_epoch = None, global_model_accuracy_last_epoch = None, best_pick_cutoff = 0.5):
  if(global_model_weights_last_epoch != None):
    epoch_results['weights'].append(global_model_weights_last_epoch)
    epoch_results['test_scores'].append(global_model_accuracy_last_epoch)

  epoch_results = sort_epoch_results(epoch_results)
  epoch_results = scale_test_scores(epoch_results)

  new_weights = []
  new_test_scores = []
  new_fl_avg_weights = []

  for index, fl_avg_weight in enumerate(epoch_results['fl_avg_weights']):
      if fl_avg_weight >= best_pick_cutoff:
          new_weights.append(epoch_results['weights'][index])
          new_test_scores.append(epoch_results['test_scores'][index])
          new_fl_avg_weights.append(fl_avg_weight)

  # Update the epoch_results dictionary with the new lists
  epoch_results['weights'] = new_weights
  epoch_results['test_scores'] = new_test_scores
  epoch_results['fl_avg_weights'] = new_fl_avg_weights

  print(epoch_results)
  weighted_average_weights_curr_epoch = calculate_weighted_average(epoch_results['weights'], epoch_results['fl_avg_weights'])
  return weighted_average_weights_curr_epoch

def simple_averaging(epoch_results, global_model_weights_last_epoch = None, global_model_accuracy_last_epoch = None):
  if(global_model_weights_last_epoch != None):
    epoch_results['weights'].append(global_model_weights_last_epoch)
    epoch_results['test_scores'].append(global_model_accuracy_last_epoch)

  epoch_weights = epoch_results['weights']
  averages = []
  # Iterate through the columns (i.e., elements at the same position) of the arrays
  for col in range(len(epoch_weights[0])):
      # Initialize a variable to store the sum of elements at the same position
      col_sum = 0
      for row in range(len(epoch_weights)):
          col_sum += epoch_weights[row][col]

      # Calculate the average for this column and append it to the averages list
      col_avg = col_sum / len(epoch_weights)
      averages.append(col_avg)

  return averages

# Streamlit Sidebar Parameters
st.sidebar.header("Federated Learning Parameters")
num_clients = st.sidebar.slider("Number of Clients", 1, 10, 5)
num_epochs = st.sidebar.slider("Number of Epochs", 1, 10, 5)
max_train_iterations = st.sidebar.slider("Max Training Iterations", 1, 50, 15)
samples_per_epoch = st.sidebar.slider("Samples per Epoch", 100, 1000, 200)
aggregation_method = st.sidebar.selectbox("Aggregation Method", ["weighted_average", "weighted_average_best_pick", "simple_averaging"])

# Train Function with Callback
def train(data, model=None):
    if model is None:
        num_features = len(data[0]["sequence"])
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        optimizer = COBYLA(maxiter=max_train_iterations)
        vqc_model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=BackendSampler(backend=BasicProvider().get_backend("basic_simulator")),
            warm_start=True
        )
        model = vqc_model

    train_sequences = [data_point["sequence"] for data_point in data]
    train_labels = [data_point["label"] for data_point in data]
    train_sequences = np.array(train_sequences)
    train_labels = np.array(train_labels)

    model.fit(train_sequences, train_labels)
    train_score_q = model.score(train_sequences, train_labels)
    test_score_q = model.score(test_sequences, test_labels)
    return train_score_q, test_score_q, model

# Main Training Loop
if st.sidebar.button("Start Federated Training"):
    st.write("### Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Initialize clients and global model
    clients = split_dataset(num_clients, num_epochs, samples_per_epoch)
    global_model_weights = []
    global_model_metrics = []

    for epoch in range(num_epochs):
        status_text.text(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar.progress((epoch + 1) / num_epochs)

        epoch_results = {'weights': [], 'test_scores': []}

        # Train each client
        for client in clients:
            client_data = client.data[epoch]
            train_score_q, test_score_q, model = train(client_data, client.primary_model)
            client.models.append(model)
            client.test_scores.append(test_score_q)
            client.train_scores.append(train_score_q)
            epoch_results['weights'].append(model.weights)
            epoch_results['test_scores'].append(test_score_q)

        # Aggregate models
        sorted_results = sort_epoch_results(epoch_results)
        scaled_results = scale_test_scores(sorted_results)
        weighted_avg_weights = calculate_weighted_average(scaled_results['weights'], scaled_results['fl_avg_weights'])
        global_model_weights.append(weighted_avg_weights)

        # Update clients with the new global model
        num_features = len(test_sequences[0])
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        optimizer = COBYLA(maxiter=max_train_iterations)
        global_model = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=BackendSampler(backend=BasicProvider().get_backend("basic_simulator")),
            initial_point=weighted_avg_weights
        )

        for client in clients:
            client.primary_model = global_model

        # Calculate metrics
        predictions = global_model.predict(test_sequences)
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')

        # Display metrics
        st.write(f"**Epoch {epoch + 1} Results**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy:.2f}")
        col2.metric("Precision", f"{precision:.2f}")
        col3.metric("Recall", f"{recall:.2f}")
        col4.metric("F1 Score", f"{f1:.2f}")

    st.success("Training Complete!")
