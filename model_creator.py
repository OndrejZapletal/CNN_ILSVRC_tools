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
        model_name = "new_config_2"
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


def get_config_for_trained(name):
    """Return config for already trained model."""
    trained_config = """{
    "epochs" : 100,
    "loss" : "categorical_crossentropy",
    "metrics" : "accuracy",
    "name" : "%s",
    "source" : "../datasets/image_net_prepared.h5",
    "steps_per_epoch" : 1000,
    "test_batch_size" : 10,
    "train_batch_size" : 50,
    "validation_steps" : 1000,
    "verbose" : 1,
    "weights" : "../models/model_%s_adam_weights.h5"
}""" % (name, name)

    return trained_config

def get_config_for_new(name):
    """Return config for new model."""
    new_config = """{
    "epochs" : 1,
    "loss" : "categorical_crossentropy",
    "metrics" : "accuracy",
    "name" : "%s",
    "source" : "../datasets/image_net_prepared.h5",
    "steps_per_epoch" : 100,
    "test_batch_size" : 10,
    "train_batch_size" : 50,
    "validation_steps" : 1000,
    "verbose" : 1,
    "weights" : ""
}""" % name

    return new_config


def save_model_generator_to_file(model, name):
    try:
        with open("../models/model_%s_parameters.json" % name, 'w') as parameters, \
             open("../models/model_%s_configuration.json" % name, 'w') as configuration:
            parameters.write(model.to_json())
            # configuration.write(get_config_for_new(name))
            configuration.write(get_config_for_trained(name))
            print("model %s saved" % name)
        return True
    except Exception as general_exception:
        print(general_exception, sys.exc_info())
        return False


def create_model_1(shape=(256, 256, 3), num_classes=40):
    """Creates appropriate model for the input data."""
    model = Sequential()
    layer1 = Conv2D(96, (11, 11), strides=(4), padding='same', input_shape=shape)

    model.add(layer1)
    model.add(Activation('relu'))

    layer2 = Conv2D(256, (5, 5), padding='same')
    model.add(layer2)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    layer3 = Conv2D(384, (3, 3), padding='same')
    model.add(layer3)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    layer4 = Conv2D(384, (3, 3), padding='same')
    model.add(layer4)
    model.add(Activation('relu'))

    layer5 = Conv2D(256, (3, 3), padding='same')
    model.add(layer5)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

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
