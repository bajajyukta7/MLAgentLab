import os
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import git
import shutil

# Clone the dataset
repo_url = 'https://github.com/laxmimerit/dog-cat-full-dataset'
repo_dir = 'dog-cat-full-dataset'
if not os.path.exists(repo_dir):
    git.Repo.clone_from(repo_url, repo_dir)

# Prepare directories
data_dir = os.path.join(repo_dir, 'data')
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Utility for loading and splitting data
def load_and_split_data(data_dir, sample_size=0.01, seed=42):
    data_gen = ImageDataGenerator(rescale=1./255)
    data_flow = data_gen.flow_from_directory(data_dir, target_size=(150, 150), batch_size=1, class_mode='binary', shuffle=True, seed=seed)
    
    file_paths = []
    labels = []
    total_items = int(len(data_flow.filenames) * sample_size)
    for i in range(total_items):
        file_paths.append(os.path.join(data_dir, data_flow.filenames[i]))
        labels.append(data_flow.classes[i])
    
    return train_test_split(file_paths, labels, test_size=0.7, random_state=seed)

# Load and split the data
train_files, test_files, train_labels, test_labels = load_and_split_data(train_dir)
val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels, test_size=0.5, random_state=42)

def create_dataset(files, labels):
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    def process_file(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [150, 150])
        img /= 255.0
        return img, label
    
    dataset = dataset.map(process_file)
    return dataset

# Create TensorFlow datasets
train_dataset = create_dataset(train_files, train_labels).batch(32)
val_dataset = create_dataset(val_files, val_labels).batch(32)
test_dataset = create_dataset(test_files, test_labels).batch(32)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)

print(f'Test accuracy: {test_acc}')

# Save the model
model.save("cat_dog_classifier.h5")

print("Model saved as cat_dog_classifier.h5")

# Streamlit download button generation (placeholder)
try:
    import streamlit as st
    with open("cat_dog_classifier.h5", "rb") as f:
        st.download_button("Download Trained Model", f, file_name="cat_dog_classifier.h5")
except ImportError:
    print("Streamlit is not installed, cannot generate download button.")