from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import os
from sklearn.model_selection import train_test_split

import keras.backend as K
K.set_image_data_format('channels_first')
print(K.image_data_format)


def generator_model():
    model = Sequential()
    model.add(Dense(1024, input_dim=100))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(
        64, (5, 5),
        padding='same',
        input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
              j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


def changed_sized_train_data(x_train, y_train, N):
    ratio = N / x_train.shape[0]
    _, X_train, _, y_train = train_test_split(x_train, y_train,
                                              test_size=ratio, random_state=42)
    return X_train, y_train


def train(args):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    train_size = args.train_size
    digit = args.digit

    # Make dir for saving the images during training
    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    (X_train, y_train), (_, _) = mnist.load_data()

    # Extract only specific digit number
    if digit != -1:
        idx = np.where(y_train == digit)[0]
        X_train = X_train[idx]
        y_train = y_train[idx]

    # change train size
    org_tr_size = X_train.shape[0]
    X_train, y_train = changed_sized_train_data(X_train, y_train, train_size)
    reduced_tr_size = X_train.shape[0]
    print('Use {0} of {1} for train'.format(reduced_tr_size, org_tr_size))

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])

    # Save original images
    image = combine_images(X_train)
    Image.fromarray(image.astype(np.uint8)).save(output_fold + '/' + "original_image.png")

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))

    d_loss_ll = []
    g_loss_ll = []
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            generated_images = generator.predict(noise, verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            # print("batch %d d_loss : %f" % (index, d_loss))
            d_loss_l.append(d_loss)

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)

            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            # print("batch %d g_loss : %f" % (index, g_loss))
            g_loss_l.append(g_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            image = combine_images(generated_images)
            image = image * 127.5 + 127.5
            Image.fromarray(image.astype(np.uint8)).save(
                output_fold + '/' +
                str(epoch) + "_" + str(index) + ".png")

            generator.save_weights(output_fold + '/' + 'generator', True)
            discriminator.save_weights(output_fold + '/' + 'discriminator', True)

        d_loss_ll.append(d_loss_l)
        g_loss_ll.append(g_loss_l)

    # Save by binary format
    # loss_d = {'d_loss': d_loss_ll, 'g_loss': g_loss_ll}
    # np.save('loss_d', loss_d)
    # Save text
    np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)


def generate(args):
    BATCH_SIZE = args.batch_size
    nice = args.nice
    output_fold = args.output_fold
    digit = args.digit

    # Make dir for saving the images during training
    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights(output_fold + '/' +'generator')

    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights(output_fold + '/' +'discriminator')
        noise = np.zeros((BATCH_SIZE * 20, 100))

        for i in range(BATCH_SIZE * 20):
            noise[i, :] = np.random.uniform(-1, 1, 100)

        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)

        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]

        image = combine_images(nice_images)
        all_images = nice_images
    else:
        noise = np.zeros((BATCH_SIZE, 100))

        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)

        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
        all_images = generated_images

    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(output_fold + '/' + "generated_image.png")

    # Save all images
    all_images = all_images * 127.5 + 127.5
    for ix, image in enumerate(all_images):
        Image.fromarray(image.astype(np.uint8)).save(output_fold + '/' + '{0}_{1}.png'.format(digit, ix))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--output_fold", default="out")
    parser.add_argument("--train_size", type=int, default=40)
    parser.add_argument("--digit", type=int, default=-1)
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate(args)
