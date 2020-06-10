import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from keras.utils import to_categorical
from keras import callbacks, optimizers, layers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


# Build Model
def build_model(bs):
    inputs = layers.Input(batch_shape=(bs, 28, 28, 1))
    x = layers.Reshape(target_shape=[784])(inputs)
    x = layers.Dense(units=784, activation='relu')(x)
    x = layers.Dense(units=300, activation='relu')(x)
    outputs = layers.Dense(units=10, activation='sigmoid')(x)
    return Model(inputs=[inputs], outputs=[outputs])


# Load data
def load_mnist(data_path):
    if os.path.exists(data_path):  # Load from local drive
        f = np.load(data_path)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    else:  # Load from web
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


# Data generator
def train_generator(x, y, bs, shift_fraction=0.):
    # Data augmentation
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction, height_shift_range=shift_fraction)
    generator = train_datagen.flow(x, y, batch_size=bs)
    while 1:
        x_batch, y_batch = generator.next()
        yield [x_batch, y_batch]


# Training
def train(model, data):
    # unpacking the data
    (x_train, y_train), (_, _) = data

    # callbacks
    callbacks_list = [callbacks.TensorBoard(log_dir='./logs', write_graph=True, write_images=False)]

    # compile the model
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    model.compile(optimizer=optimizers.Adam(lr=lr), loss=['mse'], metrics={'accuracy': 'accuracy'}, options=run_options,
                  run_metadata=run_metadata)

    # Training
    model.fit_generator(generator=train_generator(x_train, y_train, batch_size), steps_per_epoch=train_steps,
                        epochs=epochs, callbacks=callbacks_list)

    # Output timeline.json
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    print('timeline.json has been saved!')


if __name__ == '__main__':
    batch_size = 100
    lr = 0.001
    epochs = 1
    train_steps = 10  # Set to 10 just to output timeline.json

    mnist_data = load_mnist(data_path='./mnist.npz')  # Load data
    mnist_model = build_model(batch_size)  # Build model
    train(mnist_model, mnist_data)  # Train and output timeline.json
