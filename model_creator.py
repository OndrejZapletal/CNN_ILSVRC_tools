#!/usr/bin/env python3
"""This is test of keras library."""

import sys
import uuid

import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential


def main():
    """main function"""
    # create_dataset()
    model_protypes = []
    model_protypes.append(create_model_1)
    save_models(model_protypes)


def create_dataset():
    print("saving cifar10 dataset into ../datest/ folder")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    np.save("../dataset/x_train.npy", x_train)
    np.save("../dataset/y_train.npy", y_train)
    np.save("../dataset/x_test.npy", x_test)
    np.save("../dataset/y_test.npy", y_test)


def save_models(models):
    """Save models to JSON"""
    # name = (str(uuid.uuid4())[:8]).upper()
    for i, model in enumerate(models, start=2):
        # model_name = "%s_%s" % (i, name)
        model_name = "sanity_check_%s" % i
        save_model_to_file(model(), model_name)


def save_model_to_file(model, name):
    """Save model to json."""
    config = """{
    "name" : "%s",
    "loss" : "categorical_crossentropy",
    "metrics" : "accuracy",
    "batch_size" : 128,
    "epochs" : 20,
    "verbose" : 1,
    "validation_split": 0.3,
    "source" : "",
    "nb_classes" : 10,
    "max_val" : 255
}""" % name
    try:
        with open("../models/model_%s_parameters.json" % name, 'w') as json_file:
            json_file.write(model.to_json())
        with open("../models/model_%s_configuration.json" % name, 'w') as json_file:
            json_file.write(config)
            print("model %s saved" % name)
        return True
    except Exception as general_exception:
        print(general_exception, sys.exc_info())
        return False


# def create_model_1(shape=(32, 32, 3), num_classes=10):
#     """Creates apropriate model for the input data."""
#     model = Sequential()

#     model.add(Conv2D(64, (3, 3), padding='same', input_shape=shape))
#     model.add(Activation('relu'))

#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())

#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))

#     return model


# def create_model_2(shape=(32, 32, 3), num_classes=10):
#     """Creates apropriate model for the input data."""
#     model = Sequential()

#     model.add(Conv2D(64, (3, 3), padding='same', input_shape=shape))
#     model.add(Activation('relu'))

#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())

#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))

#     return model


# def create_model_3(shape=(32, 32, 3), num_classes=10):
#     """Creates apropriate model for the input data."""
#     model = Sequential()

#     model.add(Conv2D(64, (3, 3), padding='same', input_shape=shape))
#     model.add(Activation('relu'))

#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(Conv2D(256, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))

#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())

#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))

#     return model


def create_model_1(shape=(32, 32, 3), num_classes=10):
    """Creates appropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(128, (3, 3), padding='same', input_shape=shape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (4, 4)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def create_model_2(shape=(32, 32, 3), num_classes=10):
    """Creates appropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(128, (4, 4), padding='same', input_shape=shape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (4, 4)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (4, 4)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (4, 4)))
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

if __name__ == "__main__":
    main()
