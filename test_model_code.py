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


# Define paths
data_dir = os.path.join(local_path, 'data')

# Prepare image data generators
batch_size = 32
img_height = 150
img_width = 150


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