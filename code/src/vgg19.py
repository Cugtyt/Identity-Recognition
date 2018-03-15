"""Use vgg19 to do transfer learning."""
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten


def vgg19(input_shape: tuple, classes: int):
    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(classes, activation='softmax')
    ])
    conv_base.trainable = False
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def vgg16(input_shape: tuple, classes: int):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(classes, activation='softmax')
    ])
    conv_base.trainable = False
    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model