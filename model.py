import numpy as np
import os
import cv2
from tqdm import tqdm
import pickle

from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# Load the dataset
DATADIR = "dataset"
CATEGORIES = ["int", "sum", "product"]
training_data = []

# Model configuration
IMG_SIZE = 50
no_classes = 3

# Check the Data
# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR, category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
#         print(img_array)
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!
#         break
#     break


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification  (0, 1 or 2)
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                raise RuntimeError(f"Exception happened: {e}")
    return training_data


create_training_data()

# Create dataset model
X = []
y = []
# y = np.array(y)

for features, label in training_data:
    X.append(features)  # X contains the images
    y.append(label)     # y contains the classes
    # np.array((y, label))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # create an np array
y = np.array(y).reshape(-1, 1)

print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {y.shape}")

print(f"Type of X: {type(X)}")

# Save the data set ina pickle file
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# Load data set
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# Rescale Data
X = X/255.0

# Configure dataset for performace


# Define the Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# Now we have a 64x2 Model (64 neurons in the input and 2 layers)

model.add(Flatten())  # Flatten the data because Convolution is 2D and Dense is 1D dataset layer
model.add(Dense(32, activation='relu'))

# Output layer
model.add(Dense(no_classes, activation='softmax'))

# Compile the Model
model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

# Fit data to model
# The Batch size and validation split depend on the size of the dataset
model.fit(X, y, batch_size=1, epochs=3, verbose=1, validation_split=0.05)

# Evaluate the model
score = model.evaluate(X, y, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# save the model
filepath = '.'
save_model(model, filepath)


