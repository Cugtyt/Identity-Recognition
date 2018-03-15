"""Train model."""
from .data_prepare import *
from .vanilla_cnn import *
from .analyse_utils import *
from .vgg19 import *
from .resnet import *
from keras.preprocessing.image import ImageDataGenerator


def train_asian_vanilla(plot: bool = False,
                        use_tensorboard: bool = False,
                        log_dir: str = None):
    train_data, test_data, train_label, test_label = avg_float_asian()
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    train_label, val_label = train_test_split(train_label, test_size=0.2, random_state=42)
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=
        False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=
        0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=
        0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=
        0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(train_data)

    model = vanilla_cnn(
        input_shape=train_data[0].shape, classes=train_label.max() + 1)

    if use_tensorboard:
        history = model.fit_generator(
            datagen.flow(train_data, train_label, batch_size=20),
            epochs=80,
            validation_data=(val_data, val_label),
            callbacks=get_tensorboard_config(log_dir))
    else:
        history = model.fit_generator(
            datagen.flow(train_data, train_label, batch_size=20),
            epochs=80,
            validation_data=(val_data, val_label))

    model.save('./model/asian_vanilla.h5')

    if plot:
        plot_history(history)


def train_asian_vgg(plot: bool = False,
                    use_tensorboard: bool = False,
                    log_dir: str = None):
    train_data, test_data, train_label, test_label = avg_float_asian()
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    train_label, val_label = train_test_split(train_label, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    datagen.fit(train_data)

    model = vgg19(input_shape=train_data[0].shape, classes=train_label.max() + 1)

    if use_tensorboard:
        history = model.fit_generator(
            datagen.flow(train_data, train_label, batch_size=20),
            epochs=80,
            validation_data=(val_data, val_label),
            callbacks=get_tensorboard_config(log_dir))
    else:
        history = model.fit_generator(
            datagen.flow(train_data, train_label, batch_size=20),
            epochs=80,
            validation_data=(val_data, val_label))

    model.save('./model/asian_vgg19.h5')

    if plot:
        plot_history(history)


def train_asian_res(plot: bool = False,
                    use_tensorboard: bool = False,
                    log_dir: str = None):
    train_data, test_data, train_label, test_label = avg_float_asian()
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    train_label, val_label = train_test_split(train_label, test_size=0.2, random_state=42)
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    datagen.fit(train_data)

    model = resnet(input_shape=train_data[0].shape, classes=train_label.max() + 1)

    if use_tensorboard:
        history = model.fit_generator(
            datagen.flow(train_data, train_label, batch_size=20),
            epochs=80,
            validation_data=(val_data, val_label),
            callbacks=get_tensorboard_config(log_dir))
    else:
        history = model.fit_generator(
            datagen.flow(train_data, train_label, batch_size=20),
            epochs=80,
            validation_data=(val_data, val_label))

    model.save('./model/asian_resnet.h5')

    if plot:
        plot_history(history)
