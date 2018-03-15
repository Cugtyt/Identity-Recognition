"""Plot module."""
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.callbacks import TensorBoard


def plot_history(history: History):
    """Plot history."""
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def get_tensorboard_config(log_dir: str):
    """Use tensorboard config."""
    return [
        TensorBoard(
            # Log files will be written at this location
            log_dir=log_dir,
            # We will record activation histograms every 1 epoch
            histogram_freq=1,
            # We will record embedding data every 1 epoch
            embeddings_freq=1,
        )
    ]