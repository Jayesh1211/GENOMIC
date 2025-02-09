import streamlit as st
import numpy as np
import time
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import BackendSampler
from qiskit.providers.basic_provider import BasicProvider
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import defaultdict
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoCodingVsIntergenomicSeqs

# Streamlit UI
st.title("Quantum Federated Learning Web App")
st.sidebar.header("Model Parameters")

# User Inputs
num_clients = st.sidebar.slider("Number of Clients", 2, 20, 10)
num_epochs = st.sidebar.slider("Number of Epochs", 1, 10, 5)
max_train_iterations = st.sidebar.slider("Max Training Iterations", 5, 50, 15)
fl_technique = st.sidebar.selectbox("Federated Learning Technique", ["Best Pick Weighted Averaging", "Simple Averaging"])

# Placeholder for training status
status_placeholder = st.empty()
backend = BasicProvider().get_backend("basic_simulator")

# Load and preprocess genomic dataset
st.write("Loading and preprocessing genomic dataset...")
train_set = DemoCodingVsIntergenomicSeqs(split='train', version=0)
test_set = DemoCodingVsIntergenomicSeqs(split='test', version=0)
data_set = train_set + test_set

word_size = 50
word_combinations = defaultdict(int)
iteration = 1
for text, _ in data_set:
    for i in range(len(text)):
        word = text[i:i + word_size]
        if word_combinations.get(word) is None:
            word_combinations[word] = iteration
            iteration += 1

np_data_set = []
for i in range(len(data_set)):
    sequence, label = data_set[i]
    sequence = sequence.strip()
    words = [sequence[i:i + word_size] for i in range(0, len(sequence), word_size)]
    int_sequence = np.array([word_combinations[word] for word in words if word in word_combinations])
    data_point = {'sequence': int_sequence, 'label': label}
    np_data_set.append(data_point)

np.random.shuffle(np_data_set)
scaler = MinMaxScaler()
sequences = np.array([item['sequence'] for item in np_data_set], dtype=object)
sequences = np.vstack(sequences)
sequences_scaled = scaler.fit_transform(sequences)
for i, item in enumerate(np_data_set):
    item['sequence'] = sequences_scaled[i]

np_train_data = np_data_set[:75000]
np_test_data = np_data_set[-25000:]

# Federated Learning Helper Functions
def best_pick_weighted_averaging(epoch_results):
    sorted_results = sorted(zip(epoch_results['weights'], epoch_results['test_scores']), key=lambda x: x[1])
    return sorted_results[-1][0]  # Picking the best-performing model's weights

def simple_averaging(epoch_results):
    epoch_weights = np.array(epoch_results['weights'])
    return np.mean(epoch_weights, axis=0)

def create_vqc_model(initial_weights=None):
    num_features = len(np_train_data[0]['sequence'])
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
    optimizer = COBYLA(maxiter=max_train_iterations)
    vqc_model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=BackendSampler(backend=backend),
        initial_point=initial_weights if initial_weights is not None else None
    )
    return vqc_model

def train_vqc_model():
    global_metrics = []
    vqc_model = create_vqc_model()
    epoch_results = {'weights': [], 'test_scores': []}
    
    train_sequences = np.array([item['sequence'] for item in np_train_data])
    train_labels = np.array([item['label'] for item in np_train_data])
    test_sequences = np.array([item['sequence'] for item in np_test_data])
    test_labels = np.array([item['label'] for item in np_test_data])
    
    for epoch in range(num_epochs):
        status_placeholder.text(f"Training Epoch {epoch + 1}/{num_epochs}...")
        start_time = time.time()
        vqc_model.fit(train_sequences, train_labels)
        end_time = time.time()
        
        accuracy = vqc_model.score(test_sequences, test_labels)
        predictions = vqc_model.predict(test_sequences)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')
        
        epoch_results['weights'].append(vqc_model.weights)
        epoch_results['test_scores'].append(accuracy)
        
        if fl_technique == "Best Pick Weighted Averaging":
            new_weights = best_pick_weighted_averaging(epoch_results)
        else:
            new_weights = simple_averaging(epoch_results)
        
        vqc_model = create_vqc_model(initial_weights=new_weights)
        global_metrics.append((accuracy, precision, recall, f1))
        st.write(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}, Time = {end_time - start_time:.2f} sec")
    
    return global_metrics

if st.button("Start Training"):
    st.write("Training Started...")
    metrics = train_vqc_model()
    st.success("Training Completed!")
