# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:57:40 2024

@author: saira
"""

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os
import csv

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model_path = 'D:\\Major Project App\\saved models\\DISEASE.h5'
loaded_model = load_model(model_path)

    

    # Define a function to preprocess the input image
def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))  # Adjust target_size based on your model's input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values (if your training data was normalized)
        return img_array

  # Load an example image for prediction
def getPrediction(filename):
    input_image_path = 'D:\\Major Project App\\static\\images\\'+filename
    input_image = preprocess_image(input_image_path)

    # Make predictions
    predictions = loaded_model.predict(input_image)

    # Depending on your model architecture, you might need additional post-processing
    # For example, if your model outputs probabilities, you might want to extract the class with the highest probability.
    predicted_class_ind = np.argmax(predictions, axis=1)
    filename = 'D:\\Major Project App\\saved models\\planedisease.csv'  # Replace with your CSV filename

    class_names = load_class_names_from_csv(filename)
    predicted_class_name = get_class_name(class_names, predicted_class_ind)
    print(f"Predicted class name: {predicted_class_name}")
    return predicted_class_name 
def load_class_names_from_csv(filename):
        class_names = []
        with open(filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                class_names.append(row[0])
        return class_names

def get_class_name(class_names, idx):
        if isinstance(idx, np.ndarray):
            idx = idx.item()  # Convert NumPy scalar array to Python scalar
        if idx >= 0 and idx < len(class_names):
            return class_names[idx]
        else:
            return "Unknown"

    # Example usage:
