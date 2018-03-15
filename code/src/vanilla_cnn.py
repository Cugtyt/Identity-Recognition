"""
vanilla CNN keras code.

Use simple ConvNet to implement face recognition algo.
"""
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def vanilla_cnn(input_shape: tuple, classes: int):
    """Implement vanilla ConvNet model."""
    model = models.Sequential([
        Conv2D(
            32, (3, 3),
            activation='relu',
            padding='same',
            input_shape=input_shape),
        #     layers.MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        #     layers.Dropout(0.5),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        #     layers.Dropout(0.5),

        #     layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        #     layers.MaxPooling2D((2, 2))),
        Conv2D(128, (3, 3), activation='relu'),
        #     layers.MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        #     layers.Dropout(0.5),
        Dense(classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model
