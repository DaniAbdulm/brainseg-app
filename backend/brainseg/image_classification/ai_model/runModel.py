import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
from keras.models import Sequential
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models import Unet
from efficientnet.tfkeras import preprocess_input as efficientnet_preprocess_input
from sklearn.model_selection import train_test_split

def encoder_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    x = layers.Concatenate()([x, skip_tensor])
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_unet(input_shape):
    inputs = tf.keras.Input(input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 16)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)

    # Bridge
    b1 = layers.Conv2D(256, (3, 3), padding='same')(p3)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)
    b1 = layers.Conv2D(256, (3, 3), padding='same')(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)

    # Decoder
    d1 = decoder_block(b1, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 16)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d3)

    model = tf.keras.Model(inputs, outputs, name='Custom_U-Net')
    return model

# Load model
input_shape = (256, 256, 3)
model = build_unet(input_shape)
model.load_weights('model1.keras')

# Load Image
img = #Image path to be added
img= cv2.resize(img ,(256, 256))
img= img/ 255
img= img[np.newaxis, :, :, :]

# Run image through the model
out_mask = model.predict(img)
binary_mask = (out_mask > 0.25).astype(np.uint8)

# Save the output image
overlay = cv2.addWeighted(img, 0.7, binary_mask, 0.3, 0)
cv2.imwrite('mri_with_tumor_overlay.png', output_img)

# Add code to save image to server

