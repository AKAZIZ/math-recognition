import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path
import cv2

# Seperate image
img_array = cv2.imread('dataset/product/15.png', cv2.IMREAD_GRAYSCALE)  # convert to array
new_array = cv2.resize(img_array, (50, 50))
samples_to_predict = np.array(new_array).reshape(-1, 50, 50, 1)

 
# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)
# plt.imshow(X[0])
# plt.show()

# samples_to_predict = np.array([X[i] for i in range(0, 30)])

filepath = "."

# Load the model
model = load_model(filepath, compile=True)

# predict
print(f"Shape of simple to predict: {samples_to_predict.shape}")
predictions = model.predict(samples_to_predict)
print(f"Prediction: {predictions}")

# Get classes from predicitons
classes = np.argmax(predictions, axis=1)
print(classes)
