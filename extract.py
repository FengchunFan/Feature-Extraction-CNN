#import the relevant Keras libraries
import os
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model

#load pretraind cnn model VGG16 from keras
model = keras.applications.VGG16(weights='imagenet', include_top=True)
model.summary()
