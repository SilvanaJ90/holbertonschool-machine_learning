#!/usr/bin/env python3
""" t builds the ResNet-50 architecture """

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed
    by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    Returns: the keras model
    """
    he_normal_init = K.initializers.he_normal(seed=0)

    input_layer = K.layers.Input(shape=(224, 224, 3))

    # Initial Convolutional Layer
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_initializer=he_normal_init)(input_layer)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Stage 1
    x = projection_block(x, (64, 64, 256), s=1)
    x = identity_block(x, (64, 64, 256))
    x = identity_block(x, (64, 64, 256))

    # Stage 2
    x = projection_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))
    x = identity_block(x, (128, 128, 512))

    # Stage 3
    x = projection_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))
    x = identity_block(x, (256, 256, 1024))

    # Stage 4
    x = projection_block(x, (512, 512, 2048))
    x = identity_block(x, (512, 512, 2048))
    x = identity_block(x, (512, 512, 2048))

    # Average Pooling and Fully Connected Layer
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(1000, activation='softmax')(x)  # Adjust the number of classes as needed

    # Create Model
    model = K.models.Model(inputs=input_layer, outputs=x)
    
    return model
