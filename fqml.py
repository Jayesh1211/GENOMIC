import streamlit as st
import tensorflow as tf
import numpy as np
from genomic_benchmarks.dataset_getters.pytorch_datasets import DemoCodingVsIntergenomicSeqs

# Load pretrained model
MODEL_PATH = "pretrained_model.h5"  # Update with actual path if different
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset
test_set = DemoCodingVsIntergenomicSeqs(split='test', version=0)

def preprocess_text(text, word_size=50):
    word_combinations = {}
    iteration = 1
    for i in range(len(text) - word_size + 1):
        word = text[i:i+word_size]
        if word not in word_combinations:
            word_combinations[word] = iteration
            iteration += 1
    return np.array(list(word_combinations.values())).reshape(-1, 1)

# Streamlit UI
st.title("Genomic Sequence Classification using Pretrained Quantum Model")

# Select a test sample
sample_index = st.slider("Select Test Sample", 0, len(test_set) - 1, 0)
text, label = test_set[sample_index]

# Preprocess input
processed_input = preprocess_text(text)
processed_input = np.expand_dims(processed_input, axis=0)  # Ensure correct shape

# Make prediction
prediction = model.predict(processed_input)
predicted_class = np.argmax(prediction)

# Display results
st.write(f"### Actual Label: {label}")
st.write(f"### Predicted Label: {predicted_class}")
st.bar_chart(prediction.flatten())

