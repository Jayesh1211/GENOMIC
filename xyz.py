import streamlit as st
import numpy as np
import time
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import BackendSampler
from qiskit.providers.basic_provider import BasicProvider
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

def best_pick_weighted_averaging(epoch_results):
    """Federated learning technique that selects the best performing models."""
    sorted_results = sorted(zip(epoch_results['weights'], epoch_results['test_scores']), key=lambda x: x[1])
    best_weights = sorted_results[-1][0]  # Picking the best-performing model's weights
    return best_weights

def simple_averaging(epoch_results):
    """Federated learning technique that averages model weights."""
    epoch_weights = np.array(epoch_results['weights'])
    return np.mean(epoch_weights, axis=0)

def train_vqc_model():
    """Trains a Variational Quantum Classifier (VQC) and returns metrics."""
    global_metrics = []
    num_features = 4  # Assuming 4 features for simplicity
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
    optimizer = COBYLA(maxiter=max_train_iterations)
    
    vqc_model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=BackendSampler(backend=backend)
    )
    
    # Simulated dataset
    train_sequences = np.random.rand(100, num_features)
    train_labels = np.random.randint(0, 2, 100)
    test_sequences = np.random.rand(30, num_features)
    test_labels = np.random.randint(0, 2, 30)
    
    epoch_results = {'weights': [], 'test_scores': []}
    
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
        
        # Apply selected FL technique
        if fl_technique == "Best Pick Weighted Averaging":
            new_weights = best_pick_weighted_averaging(epoch_results)
        else:
            new_weights = simple_averaging(epoch_results)
        
        vqc_model.weights = new_weights
        
        global_metrics.append((accuracy, precision, recall, f1))
        st.write(f"Epoch {epoch + 1}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}, Time = {end_time - start_time:.2f} sec")
    
    return global_metrics

if st.button("Start Training"):
    st.write("Training Started...")
    metrics = train_vqc_model()
    st.success("Training Completed!")
