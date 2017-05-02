#!/usr/bin/env python3
"""This is test of keras library."""

import sys

from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential


def main():
    """main function"""
    model_protypes = []
    model_protypes.append(create_model_1)
    save_models(model_protypes)


def save_models(models):
    """Save models to JSON"""
    for _, model in enumerate(models, start=1):
        # model_name = "%s_%s" % (i, name)
        model_name = "complex_model_for_imagenet"
        # save_model_to_file(model(), model_name)
        save_model_generator_to_file(model(), model_name)


def save_model_to_file(model, name):
    """Save model to json."""
    config = """{
    "name" : "%s",
    "loss" : "categorical_crossentropy",
    "metrics" : "accuracy",
    "batch_size" : 256,
    "epochs" : 300,
    "verbose" : 1,
    "validation_split": 0.3,
    "source" : "",
    "nb_classes" : 10
}""" % name
    try:
        with open("../models/model_%s_parameters.json" % name, 'w') as parameters, \
             open("../models/model_%s_configuration.json" % name, 'w') as configuration:
            parameters.write(model.to_json())
            configuration.write(config)
            print("model %s saved" % name)
        return True
    except Exception as general_exception:
        print(general_exception, sys.exc_info())
        return False


def save_model_generator_to_file(model, name):
    """Save model to json."""
    config = """{
    "epochs" : 300,
    "loss" : "categorical_crossentropy",
    "metrics" : "accuracy",
    "name" : "%s",
    "source" : "image_net_prepared.h5",
    "steps_per_epoch" : 10,
    "test_batch_size" : 10
    "train_batch_size" : 200,
    "validation_steps" : 10,
    "verbose" : 1,
}""" % name
    try:
        with open("../models/model_%s_parameters.json" % name, 'w') as parameters, \
             open("../models/model_%s_configuration.json" % name, 'w') as configuration:
            parameters.write(model.to_json())
            configuration.write(config)
            print("model %s saved" % name)
        return True
    except Exception as general_exception:
        print(general_exception, sys.exc_info())
        return False


def create_model_1(shape=(256, 256, 3), num_classes=40):
    """Creates appropriate model for the input data."""
    model = Sequential()

    model.add(Conv2D(256, (5, 5), padding='same', input_shape=shape))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


if __name__ == "__main__":
    main()
