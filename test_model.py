import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
import pickle
from pathlib import Path

# samples_to_predict = mpimg.imread('resized.png')
# samples_to_predict = np.array(samples_to_predict)


pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
plt.imshow(X[0])
plt.show()

samples_to_predict = np.array([X[1], X[11], X[21]])

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
