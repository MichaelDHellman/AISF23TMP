import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def double_conv_block(x, n_filters):

    #perform 2d convolution with relu activation, twice
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer= "he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer= "he_normal")(x)
    
    return x

def downsampler(x, n_filters):
    #Here f is our feature, and p is our layer for the model
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p

def upsampler(x, conv_features, n_filters):
    #Upsample step
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    #concatenate upsampled x
    x = layers.concatenate([x, conv_features])
    #dropout to mitigate overfitting
    x = layers.Dropout(0.3)(x)
    #Perform double conv operation
    double_conv_block(x, n_filters)

    return x

def Unet():
    inputs = layers.Input(shape=(128,128,3))

    f1, p1 = downsampler(inputs, 64)
    f2, p2 = downsampler(p1, 128)
    f3, p3 = downsampler(p2, 256)
    f4, p4 = downsampler(p3, 512)

    bedrock = double_conv_block(p4, 1024)

    u6 = upsampler(bedrock, f4, 512)
    u7 = upsampler(u6, f3, 256)
    u8 = upsampler(u7, f2, 128)
    u9 = upsampler(u8, f1, 64)

    outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

    model = tf.keras.Model(inputs, outputs, name="UNET")

    return model