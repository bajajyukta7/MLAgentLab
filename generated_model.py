import os
import subprocess
import zipfile
import requests
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Clone the repository and extract the dataset
repo_url = 'https://github.com/laxmimerit/dog-cat-full-dataset'
local_path = 'dog-cat-full-dataset'
if not os.path.exists(local_path):
    subprocess.run(['git', 'clone', repo_url, local_path])

# Define paths
data_dir = os.path.join(local_path, 'data')

# Prepare image data generators
batch_size = 32
img_height = 150
img_width = 150

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.3)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

# Train the model
epochs = 20
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print(f'Validation Accuracy: {evaluation[1]*100:.2f}%')

# Load the best model for testing
model = tf.keras.models.load_model('best_model.h5')

# If test data is separate, similar loading and evaluation process as validation can be done
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
)

# Evaluate the model on test data
evaluation = model.evaluate(test_generator)
print(f'Test Accuracy: {evaluation[1]*100:.2f}%')

# Assuming the data is split perfectly into required sets for simplicity