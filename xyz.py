# Import necessary libraries
import streamlit as st
import numpy as np
from collections import defaultdict
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoCodingVsIntergenomicSeqs
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import BackendSampler
from functools import partial
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load and preprocess data
def load_data():
    # Force download the dataset
    test_set = DemoCodingVsIntergenomicSeqs(split='test', version=0, force_download=True)
    train_set = DemoCodingVsIntergenomicSeqs(split='train', version=0, force_download=True)
    data_set = train_set + test_set
    return data_set

# Function to preprocess the dataset
def preprocess_data(data_set, word_size=50):
    word_combinations = defaultdict(int)
    iteration = 1
    for text, _ in data_set:
        for i in range(len(text)):
            word = text[i:i+word_size]
            if word_combinations.get(word) is None:
                word_combinations[word] = iteration
                iteration += 1

    np_data_set = []
    for i in range(len(data_set)):
        sequence, label = data_set[i]
        sequence = sequence.strip()
        words = [sequence[i:i + word_size] for i in range(0, len(sequence), word_size)]
        int_sequence = np.array([word_combinations[word] for word in words])
        data_point = {'sequence': int_sequence, 'label': label}
        np_data_set.append(data_point)

    # Shuffle and scale the dataset
    np.random.shuffle(np_data_set)
    sequences = np.array([item['sequence'] for item in np_data_set])
    sequences = np.vstack(sequences)
    scaler = MinMaxScaler()
    sequences_scaled = scaler.fit_transform(sequences)

    for i, item in enumerate(np_data_set):
        item['sequence'] = sequences_scaled[i]

    return np_data_set

# Function to train the model
def train_model(data, num_features, backend):
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
    optimizer = COBYLA(maxiter=15)
    vqc_model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, sampler=BackendSampler(backend=backend))
    
    train_sequences = np.array([data_point["sequence"] for data_point in data])
    train_labels = np.array([data_point["label"] for data_point in data])
    
    vqc_model.fit(train_sequences, train_labels)
    return vqc_model

# Function to get metrics
def get_metrics(model, test_sequences, test_labels):
    predictions = model.predict(test_sequences)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    return accuracy, precision, recall, f1

# Weight aggregation functions
def simple_averaging(epoch_results):
    epoch_weights = epoch_results['weights']
    averages = []
    for col in range(len(epoch_weights[0])):
        col_sum = sum(epoch_weights[row][col] for row in range(len(epoch_weights)))
        col_avg = col_sum / len(epoch_weights)
        averages.append(col_avg)
    return averages

def weighted_average(epoch_results):
    weights = epoch_results['weights']
    test_scores = epoch_results['test_scores']
    fl_avg_weights = [score / sum(test_scores) for score in test_scores]
    weighted_sum_weights = []
    for index in range(len(weights[0])):
        weighted_sum_weights.append(sum(weights_array[index] * avg_weight for weights_array, avg_weight in zip(weights, fl_avg_weights)) / sum(fl_avg_weights))
    return weighted_sum_weights

# Streamlit interface
st.title("Genomic Benchmarking with Qiskit")
st.sidebar.header("User   Input")

# Load data
data_set = load_data()
np_data_set = preprocess_data(data_set)

# User inputs
num_clients = st.sidebar.number_input("Number of Clients", min_value=1, max_value=100, value=10)
num_epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=20, value=5)

# Backend selection
backend_options = ["basic_simulator", "aer_simulator"]
backend_choice = st.sidebar.selectbox("Select Backend", backend_options)
backend = BasicProvider().get_backend(backend_choice)

# Split data into training and testing sets
np_train_data = np_data_set[:75000]
np_test_data = np_data_set[-25000:]

test_sequences = np.array([data_point["sequence"] for data_point in np_test_data])
test_labels = np.array([data_point["label"] for data_point in np_test_data])

# Train model button
if st.sidebar.button("Train Model"):
    global_model_weights = []
    global_model_metrics = []

    for epoch in range(num_epochs):
        epoch_results = {
            'weights': [],
            'test_scores': []
        }
        for client_index in range(num_clients):
            model = train_model(np_train_data, len(np_train_data[0]["sequence"]), backend)
            epoch_results['weights'].append(model.weights)
            accuracy, precision, recall, f1 = get_metrics(model, test_sequences, test_labels)
            epoch_results['test_scores'].append(accuracy)

        if epoch == 0:
            new_global_weights = simple_averaging(epoch_results)
        else:
            new_global_weights = weighted_average(epoch_results)

        global_model_weights.append(new_global_weights)

        # Get metrics for the global model
        new_model_with_global_weights = create_model_with_weights(new_global_weights)
        accuracy, precision, recall, f1 = get_metrics(new_model_with_global_weights, test_sequences, test_labels)
        global_model_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        st.write(f"Epoch {epoch + 1}/{num_epochs} - Global Model Accuracy: {accuracy:.2f}")

# Display first few samples of the dataset
st.write("First 5 Samples of Encoded Data:")
st.write(np_data_set[:5])
