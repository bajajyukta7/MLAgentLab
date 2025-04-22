import os
import git
import subprocess
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
import streamlit as st

# Clone the GitHub repository
repo_url = 'https://github.com/laxmimerit/dog-cat-full-dataset'
repo_dir = 'dog-cat-full-dataset'

if not os.path.exists(repo_dir):
    git.Repo.clone_from(repo_url, repo_dir)

# Set dataset paths
train_dir = os.path.join(repo_dir, 'data/train')
test_dir = os.path.join(repo_dir, 'data/test')
sample_dir = 'sample_data'

# Create a sample directory if it doesn't exist
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Function to sample 1% of the data
def sample_data(src_dir, dest_dir, sample_fraction=0.01):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for category in ['cats', 'dogs']:
        src_category_dir = os.path.join(src_dir, category)
        dest_category_dir = os.path.join(dest_dir, category)
        if not os.path.exists(dest_category_dir):
            os.makedirs(dest_category_dir)
        files = os.listdir(src_category_dir)
        sample_size = int(len(files) * sample_fraction)
        sample_files = np.random.choice(files, sample_size, replace=False)
        for file in sample_files:
            shutil.copy(os.path.join(src_category_dir, file), os.path.join(dest_category_dir, file))

# Sample 1% of the train and test data
sample_train_dir = os.path.join(sample_dir, 'train')
sample_test_dir = os.path.join(sample_dir, 'test')
sample_data(train_dir, sample_train_dir)
sample_data(test_dir, sample_test_dir)

# Define the ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)  # 70% training, 30% validation
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the training, validation, and testing data generators
train_generator = train_datagen.flow_from_directory(sample_train_dir, target_size=(150, 150), batch_size=20, class_mode='binary', subset='training')
validation_generator = train_datagen.flow_from_directory(sample_train_dir, target_size=(150, 150), batch_size=20, class_mode='binary', subset='validation')
test_generator = test_datagen.flow_from_directory(sample_test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Train the model
history = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, validation_data=validation_generator, validation_steps=len(validation_generator))

# Print training history
print("Training history:", history.history)

# Save the model
model_name = 'cat_dog_classifier.h5'
model.save(model_name)

# Streamlit download button
with open(model_name, "rb") as f:
    st.download_button("Download Trained Model", f, file_name=model_name)