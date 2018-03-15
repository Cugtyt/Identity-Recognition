"""Use ResNet to do transfer learning."""
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten


def resnet(input_shape: tuple, classes: int):
    conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
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