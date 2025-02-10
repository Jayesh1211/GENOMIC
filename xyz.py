import streamlit as st
import time
import warnings
import pandas as pd
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.primitives import BackendSampler
from qiskit.providers.basic_provider import BasicProvider
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from functools import partial

# Suppress warnings
warnings.simplefilter("ignore", FutureWarning)

# Page config
st.set_page_config(page_title="Quantum Federated Learning", layout="wide")
st.title("Quantum Federated Learning Dashboard")

# Debug mode toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Sidebar configuration
st.sidebar.header("Configuration")
num_clients = st.sidebar.slider("Number of Clients", min_value=1, max_value=5, value=2)
num_epochs = st.sidebar.slider("Number of Epochs", min_value=1, max_value=5, value=2)
max_train_iterations = st.sidebar.slider("Max Training Iterations", min_value=5, max_value=30, value=15)
samples_per_epoch = st.sidebar.slider("Samples per Epoch", min_value=50, max_value=500, value=200)

# Global configurations
fl_avg_weight_range = [0.1, 1]

# Aggregation technique selection
aggregation_technique = st.sidebar.selectbox(
    "Select Aggregation Technique",
    ["Simple Averaging", "Weighted Averaging", "Best Pick Weighted Averaging"]
)

# File uploader
uploaded_file = st.file_uploader("Upload preprocessed CSV file", type=['csv'])

if uploaded_file is not None:
    df_preprocessed = pd.read_csv(uploaded_file)
    
    # Data preprocessing
    @st.cache_data
    def preprocess_data(df):
        df["sequence"] = df["sequence"].apply(lambda x: np.array(eval(x)))
        np_data_set = [{"sequence": row["sequence"], "label": row["label"]} 
                      for _, row in df.iterrows()]
        return np_data_set

    with st.spinner("Processing data..."):
        np_data_set = preprocess_data(df_preprocessed)
        
    # Data splitting
    train_size = int(len(np_data_set) * 0.75)
    np_train_data = np_data_set[:train_size]
    np_test_data = np_data_set[train_size:]
    
    if debug_mode:
        st.write(f"Length of training data: {len(np_train_data)}")
        st.write(f"Length of test data: {len(np_test_data)}")
    
    # Extract test data
    test_sequences = np.array([data_point["sequence"] for data_point in np_test_data])
    test_labels = np.array([data_point["label"] for data_point in np_test_data])
    
    # Initialize backend
    backend = BasicProvider().get_backend("basic_simulator")
    
    # Client class (exactly as in original)
    class Client:
        def __init__(self, data):
            self.models = []
            self.primary_model = None
            self.data = data
            self.test_scores = []
            self.train_scores = []

    def sort_epoch_results(epoch_results):
        """Sort epoch results by test scores"""
        pairs = zip(epoch_results['weights'], epoch_results['test_scores'])
        sorted_pairs = sorted(pairs, key=lambda x: x[1])
        sorted_weights, sorted_test_scores = zip(*sorted_pairs)
        return {
            'weights': list(sorted_weights),
            'test_scores': list(sorted_test_scores)
        }

    def scale_test_scores(sorted_epoch_results):
        """Scale test scores using global weight range"""
        min_test_score = sorted_epoch_results['test_scores'][0]
        max_test_score = sorted_epoch_results['test_scores'][-1]
        min_weight, max_weight = fl_avg_weight_range
        scaled_weights = [
            min_weight + (max_weight - min_weight) * (test_score - min_test_score) / (max_test_score - min_test_score)
            for test_score in sorted_epoch_results['test_scores']
        ]
        sorted_epoch_results['fl_avg_weights'] = scaled_weights
        return sorted_epoch_results

    def calculate_weighted_average(model_weights, fl_avg_weights):
        """Calculate weighted average of model weights"""
        weighted_sum_weights = []
        for index in range(len(model_weights[0])):
            weighted_sum_weights.append(0)
            weighted_sum_weights[index] = sum([(weights_array[index] * avg_weight) 
                                            for weights_array, avg_weight in zip(model_weights, fl_avg_weights)])/sum(fl_avg_weights)
        return weighted_sum_weights

    def simple_averaging(epoch_results, global_weights=None, global_accuracy=None):
        """Simple averaging of weights"""
        if global_weights is not None:
            epoch_results['weights'].append(global_weights)
            epoch_results['test_scores'].append(global_accuracy)
        
        epoch_weights = epoch_results['weights']
        averages = []
        for col in range(len(epoch_weights[0])):
            col_sum = 0
            for row in range(len(epoch_weights)):
                col_sum += epoch_weights[row][col]
            col_avg = col_sum / len(epoch_weights)
            averages.append(col_avg)
        return averages

    def weighted_average(epoch_results, global_weights=None, global_accuracy=None):
        """Weighted averaging of model weights"""
        if global_weights is not None:
            epoch_results['weights'].append(global_weights)
            epoch_results['test_scores'].append(global_accuracy)
        
        sorted_results = sort_epoch_results(epoch_results)
        scaled_results = scale_test_scores(sorted_results)
        return calculate_weighted_average(scaled_results['weights'], scaled_results['fl_avg_weights'])

    def weighted_average_best_pick(epoch_results, global_weights=None, global_accuracy=None, best_pick_cutoff=0.5):
        """Weighted averaging with best pick selection"""
        if global_weights is not None:
            epoch_results['weights'].append(global_weights)
            epoch_results['test_scores'].append(global_accuracy)

        sorted_results = sort_epoch_results(epoch_results)
        scaled_results = scale_test_scores(sorted_results)
        
        # Filter weights based on cutoff
        filtered_weights = []
        filtered_fl_weights = []
        for idx, fl_weight in enumerate(scaled_results['fl_avg_weights']):
            if fl_weight >= best_pick_cutoff:
                filtered_weights.append(scaled_results['weights'][idx])
                filtered_fl_weights.append(fl_weight)
        
        if not filtered_weights:  # If no weights meet cutoff, use all weights
            return calculate_weighted_average(scaled_results['weights'], scaled_results['fl_avg_weights'])
        
        return calculate_weighted_average(filtered_weights, filtered_fl_weights)

    def getMetrics(weights, test_num=200):
        """Get model metrics using provided weights"""
        num_features = len(test_sequences[0])
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        optimizer = COBYLA(maxiter=0)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=BackendSampler(backend=backend),
            initial_point=weights
        )
        vqc.fit(test_sequences[:25], test_labels[:25])
        predictions = vqc.predict(test_sequences[:test_num])

        accuracy = vqc.score(test_sequences[:test_num], test_labels[:test_num])
        precision = precision_score(test_labels[:test_num], predictions, average='weighted')
        recall = recall_score(test_labels[:test_num], predictions, average='weighted')
        f1 = f1_score(test_labels[:test_num], predictions, average='weighted')

        return accuracy, precision, recall, f1

    def create_model_with_weights(weights):
        """Create a new model with given weights"""
        num_features = len(test_sequences[0])
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        optimizer = COBYLA(maxiter=max_train_iterations)
        vqc = VQC(
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            sampler=BackendSampler(backend=backend),
            warm_start=True,
            initial_point=weights,
            callback=partial(training_callback)
        )
        return vqc

    # Training callback
    def training_callback(weights, obj_func_eval):
        if not hasattr(st.session_state, 'itr'):
            st.session_state.itr = 0
        st.session_state.itr += 1
        if debug_mode:
            st.write(f"Training iteration: {st.session_state.itr}")
        st.session_state.progress_bar.progress(st.session_state.itr / max_train_iterations)

    # Training function
    def train(data, model=None):
        if model is None:
            num_features = len(data[0]["sequence"])
            feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
            ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
            optimizer = COBYLA(maxiter=max_train_iterations)
            model = VQC(
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
                callback=partial(training_callback),
                sampler=BackendSampler(backend=backend),
                warm_start=True
            )

        train_sequences = np.array([d["sequence"] for d in data])
        train_labels = np.array([d["label"] for d in data])

        if debug_mode:
            st.write("Train Sequences Shape:", train_sequences.shape)
            st.write("Train Labels Shape:", train_labels.shape)

        start_time = time.time()
        model.fit(train_sequences, train_labels)
        elapsed_time = time.time() - start_time

        train_score = model.score(train_sequences, train_labels)
        test_score = model.score(test_sequences, test_labels)

        return train_score, test_score, model, elapsed_time

    # Map technique names to functions
    aggregation_functions = {
        "Simple Averaging": simple_averaging,
        "Weighted Averaging": weighted_average,
        "Best Pick Weighted Averaging": weighted_average_best_pick
    }

    # Training button
    if st.button("Start Training"):
        st.session_state.progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        aggregation_func = aggregation_functions[aggregation_technique]
        
        with st.spinner("Training in progress..."):
            # Split dataset
            clients = []
            for i in range(num_clients):
                client_data = []
                for j in range(num_epochs):
                    start_idx = (i*num_epochs*samples_per_epoch)+(j*samples_per_epoch)
                    end_idx = start_idx + samples_per_epoch
                    client_data.append(np_train_data[start_idx:end_idx])
                clients.append(Client(client_data))
            
            global_metrics = []
            
            for epoch in range(num_epochs):
                status_text.text(f"Training Epoch {epoch + 1}/{num_epochs}")
                
                epoch_results = {'weights': [], 'test_scores': []}
                
                # Train each client
                for idx, client in enumerate(clients):
                    if debug_mode:
                        st.write(f"Training Client {idx + 1}")
                    
                    train_score, test_score, model, time_taken = train(
                        client.data[epoch], 
                        client.primary_model
                    )
                    
                    client.models.append(model)
                    client.test_scores.append(test_score)
                    client.train_scores.append(train_score)
                    
                    epoch_results['weights'].append(model.weights)
                    epoch_results['test_scores'].append(test_score)
                    
                    with metrics_container:
                        st.write(f"Client {idx + 1} - Epoch {epoch + 1}")
                        st.write(f"Train Score: {train_score:.4f}")
                        st.write(f"Test Score: {test_score:.4f}")
                        st.write(f"Time taken: {time_taken:.2f}s")
                        st.write("---")
                
                # Calculate global model
                if epoch == 0:
                    global_weights = aggregation_func(epoch_results)
                else:
                    global_weights = aggregation_func(
                        epoch_results,
                        global_metrics[-1]['weights'],
                        global_metrics[-1]['accuracy']
                    )
                
                # Get metrics using original getMetrics function
                accuracy, precision, recall, f1 = getMetrics(global_weights, len(test_sequences))
                
                metrics = {
                    'weights': global_weights,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                global_metrics.append(metrics)
                
                # Update clients with new global model
                new_global_model = create_model_with_weights(global_weights)
                for client in clients:
                    client.primary_model = new_global_model
                
                # Display metrics
                with metrics_container:
                    st.write(f"\nGlobal Model Metrics - Epoch {epoch + 1}")
                    st.write(f"Aggregation Technique: {aggregation_technique}")
                    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                    st.write(f"Precision: {metrics['precision']:.4f}")
                    st.write(f"Recall: {metrics['recall']:.4f}")
                    st.write(f"F1 Score: {metrics['f1_score']:.4f}")
                    st.write("="*50)
        
        status_text.text("Training Complete!")
        
        # Final metrics visualization
        st.header("Training Results")
        metrics_df = pd.DataFrame(global_metrics)
        st.line_chart(metrics_df[['accuracy', 'precision', 'recall', 'f1_score']])
        
else:
    st.warning("Please upload a preprocessed CSV file to begin training.")
