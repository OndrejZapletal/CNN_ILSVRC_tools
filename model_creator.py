#!/usr/bin/env python3
"""This is test of keras library."""

import json
import sys

from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import SGD, adam, nadam


def get_model_configuration(index):
    """Create JSON file with model configuration. """
    name = "g-force_test"

    config, settings, optimization, training, testing = {}, {}, {}, {}, {}
    config['name'] = "%s_%s" % (name, index)
    config['gpu_unit'] = get_gpu_configuration()

    optimization['loss'] = "categorical_crossentropy"
    optimization['metrics'] = ["accuracy"]
    optimization['optimizers'] = [get_optimizer("adam")]

    training['epochs'] = 30
    training['steps_per_epoch'] = 4000
    training['train_batch_size'] = 10

    testing['test_batch_size'] = 10
    testing['validation_steps'] = 100

    # Source for Dataset. If empty the CIFAR10 database is used
    settings['source'] = "../datasets/image_net_40_cat.h5"
    settings['shape'] = (256, 256, 3)
    settings['nb_classes'] = 40
    # use pre-trained weights otherwise left empty
    settings['weights'] = ""
    # settings['weights'] = "../models_%s/model_%s_%s_weights.h5" % (
    #     config['gpu_unit'], config['name'], optimization['optimizer'])
    settings['verbose'] = 1

    config['settings'] = settings
    config['optimization'] = optimization
    config['training'] = training
    config['testing'] = testing

    return config


def create_model(shape, num_classes):
    """Creates appropriate model for the input data."""
    model = Sequential()
    layer1 = Conv2D(96, (11, 11), strides=(4), padding='same', input_shape=shape)

    model.add(layer1)
    model.add(Activation('relu'))

    layer2 = Conv2D(128, (5, 5), padding='same')
    model.add(layer2)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    layer3 = Conv2D(190, (3, 3), padding='same')
    model.add(layer3)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    layer4 = Conv2D(190, (3, 3), padding='same')
    model.add(layer4)
    model.add(Activation('relu'))

    layer5 = Conv2D(128, (3, 3), padding='same')
    model.add(layer5)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model

def get_gpu_configuration():
    """Determines GPU unit configuration. Since this is essential it will exit otherwise.
    """
    gpu_config = input("select GPU type (0 - 'Nvidia 1080', 1 - 'Nvidia Tesla'):")
    if gpu_config == "0":
        return "1080"
    elif gpu_config == "1":
        return "tesla"
    else:
        print("You have to specify GPU unit.\nExiting!")
        sys.exit(0)

def get_optimizer(optimizer):
    optimizer_ref = {}
    if isinstance(optimizer, str):
        optimizer_ref['name'] = optimizer
        optimizer_ref['data'] = None
    else:
        if isinstance(optimizer, SGD):
            optimizer_ref['name'] = 'sgd'
        elif isinstance(optimizer, adam):
            optimizer_ref['name'] = 'adam'
        elif isinstance(optimizer, nadam):
            optimizer_ref['name'] = 'nadam'

        optimizer_ref['data'] = optimizer.get_config()
    return optimizer_ref


def save_model_to_file(model, config):
    try:
        with open("../models_%s/model_%s_parameters.json" % (
                config['gpu_unit'], config['name']), 'w') as parameters, \
             open("../models_%s/model_%s_configuration.json" % (
                 config['gpu_unit'], config['name']), 'w') as configuration:

            model_specification = model(config['settings']['shape'],
                                        config['settings']['nb_classes'])

            parameters.write(model_specification.to_json())
            configuration.write(json.dumps(config))
            print("model %s saved" % config['name'])
        return True
    except Exception as general_exception:
        print(general_exception, sys.exc_info())
        return False



def main():
    """main function"""
    models = []

    models.append(create_model)

    for index, model in enumerate(models):
        config = get_model_configuration(index)
        save_model_to_file(model, config)


if __name__ == "__main__":
    main()
