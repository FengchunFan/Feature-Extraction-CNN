# import the relevant Keras libraries
import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model

# load pretraind cnn model VGG16 from keras
model = keras.applications.VGG16(weights='imagenet', include_top=True)
# output summary of the VGG16 model, containing 13 convolutional layers
# set up to take fixed size (224 x 224 x 3) RGB image as input
# last layer being softmax classification layer
# output shape at each layer has "None" the first dimension
#model.summary()

import numpy as np
import matplotlib.pyplot as plt

# load_image function taken from https://colab.research.google.com/github/ml4a/ml4a/blob/master/examples/info_retrieval/image_search.ipynb
# load image function will preprocess image into correct size of 224x224x3, which is done through load_img function
# expand_dims is needed to add one more dimension in the very front, to process multiple image input
def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) 
    x = preprocess_input(x)
    return img, x

from tensorflow.keras.preprocessing import image

# load an image to test the functionality of load_image function
# output img is the image loaded from input path
# x is the Numpy array representation of the input image, it has shape of (1, height, width, channels)
img, x = load_image("C:/study/Github_Project/Feature-Extraction-CNN/Objects/octopus/image_0030.jpg")
print("shape of x: ", x.shape)
print("data type: ", x.dtype)
#print("x: ", x)
plt.imshow(img)
plt.title("sample image from dataset")
plt.show()


