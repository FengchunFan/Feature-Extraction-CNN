# import the relevant Keras libraries
import os
import keras
from keras.preprocessing import image
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model

# load pretraind cnn model VGG16 from keras
model = keras.applications.VGG16(weights='imagenet', include_top=True)
# output summary of the VGG16 model, containing 13 convolutional layers
# set up to take fixed size (224 x 224 x 3) RGB image as input
# last layer being softmax classification layer
# output shape at each layer has "None" the first dimension
# keep the weights/params constant as they are all pre-trained for us
for layer in model.layers:
    layer.trainable = False
#model.summary() 

import numpy as np
import matplotlib.pyplot as plt

# load_image function taken from https://colab.research.google.com/github/ml4a/ml4a/blob/master/examples/info_retrieval/image_search.ipynb
# load_images function modified based on the above function to take multiple input images
# load image function will preprocess image into correct size of 224x224x3, which is done through load_img function
# expand_dims is needed to add one more dimension in the very front, to process multiple image input
# x is the image array
def load_images(paths):
    images = []
    for path in paths:
        img = image.load_img(path, target_size=model.input_shape[1:3])
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append((img, x))
    return images

# load an or multiple images to test the functionality of load_image function
# output image is in format of (img, x)
# img or image[0] is the image loaded from input path
# x or image [1] is the Numpy array representation of the input image, it has shape of (1, height, width, channels)
# Predict the class of the image through VGG16 model using built in function
# input will be Numpy array x, which we extracted previously
# using [0] to output best prediction
# _ is needed because first argument being returned is label and we don't need that
paths = []
paths.append("C:/study/Research/Professor_Ucr_Jia_Chen/Feature-Extraction-CNN/Objects/garfield/image_0002.jpg")
paths.append("C:/study/Research/Professor_Ucr_Jia_Chen/Feature-Extraction-CNN/Objects/panda/image_0001.jpg")
images = load_images(paths)
for single_image in images:
    print("shape of image: ", single_image[1].shape)
    plt.imshow(single_image[0])
    plt.show()
    # Predictions is in shape (1, 1000), meaning VGG16 has 1000 built in class, each column in this array represent to likelihood of the image being in that class
    # ex. predictions[0][1] is the likelihood for input image being class 1
    predictions = model.predict(single_image[1])
    #print(predictions[0][0])
    #print("shape of predictions: ", predictions.shape) #(1, 1000)
    prediction_list = []
    for _, prediction, probability in decode_predictions(predictions)[0]:
        print("For the input image, model predicted it to be {} with probability {:.3%}".format(prediction, probability))
        prediction_list.append(prediction)
    print("The highest possible feature extraction from the image is: {}".format(prediction_list[0]))
    print()