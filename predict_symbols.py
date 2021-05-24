import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

from class_to_latexCommand_dictionary import class_to_latexCommand

filepath = "."  # file path of the model file (.bp file)

# The model is moved here, to not load the model at every time we initialize the object.
model = load_model(filepath, compile=True)  # Load the model


class SymbolPredictor:
    def __init__(self, symbol_detector):

        self.symbol_detector = symbol_detector
        self.image_of_symbol_to_predict = None
        self.np_array_of_symbols_to_predict = []
        self.symbol_classes_list = []
        self.latex_commands_list = []

    def preprocess_images(self, images):
        # img_array = cv2.imread('dataset/product/15.png', cv2.IMREAD_GRAYSCALE)  # convert to array
        # TODO: check size before resizing
        # TODO: import image size instead of using the hardcoded value
        # TODO: Why the image here still need to be grayed?
        if images is None:
            raise ValueError("images passed to preprocess_images() is None!")
        for image in images:
            print(f"Number of images to preprocess: {len(images)}")
            resized_image = cv2.resize(image, (50, 50))
            image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            np_array_image = np.array(image_gray).reshape(-1, 50, 50, 1)
            if len(np_array_image) != 1:
                raise RuntimeError(f"The image must be in Gray Scale not RGB.")
            self.np_array_of_symbols_to_predict.append(np_array_image)

    def predict_symbol_class(self):
        # predict
        if self.np_array_of_symbols_to_predict is not None:
            for symbol in self.np_array_of_symbols_to_predict:
                predictions = model.predict(symbol)
                print(f"Prediction: {predictions}")
                # Get classes from predicition
                # np.argmax() returns an array with 1 element which is the index of the max value
                symbol_class = np.argmax(predictions, axis=1)
                print(f"Symbole Class: {symbol_class}")
                self.symbol_classes_list.append(str(symbol_class[0]))
        else:
            raise ValueError("np_array_of_symbols_to_predict is None! No image has been passed to the preprocess_image() method")

    def convert_class_to_latex_command(self):
        for symbol_class in self.symbol_classes_list:
            latex_command = class_to_latexCommand[str(symbol_class)]
            self.latex_commands_list.append(latex_command)
            print(latex_command)

    def predict(self):
        self.__init__(symbol_detector=self.symbol_detector)
        self.preprocess_images(images=self.symbol_detector.detected_symbols_list)
        self.predict_symbol_class()
        self.convert_class_to_latex_command()
