# -*- coding: utf-8 -*-
"""image_classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G3T812v1TZUr2-hnLPLJ7Z4sagqIgatQ
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('cnn_image_classification_model.h5')

# Function to preprocess and predict new images
def preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(32, 32))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Example prediction on a new image (uncomment the lines below to test)
# image_path = 'path_to_new_image.jpg'
# predicted_class = predict_image(image_path)
# print(f'Predicted class: {predicted_class}')

image_path = '/content/Vidwud_faceswap.jpg'
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')

image_path = '/content/gosau-3724039.jpg'
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')