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
    name = "testing_batch_size"
    dataset = "image_net_40_cat"
    classes = 40

    config, settings, optimization, training, testing = {}, {}, {}, {}, {}
    config['gpu_unit'] = get_gpu_configuration()


    optimization['loss'] = "categorical_crossentropy"
    optimization['metrics'] = ["accuracy"]
    optimization['optimizers'] = [get_optimizer("adam"),
                                  get_optimizer("nadam"),
                                  get_optimizer("sgd")]
    training['epochs'] = 30
    training['steps_per_epoch'] = 200
    training['train_batch_size'] = 20

    testing['validation_steps'] = 180
    testing['test_batch_size'] = 10

    # Source for Dataset. If empty the CIFAR10 database is used
    settings['source'] = "../datasets/%s.h5" % dataset
    settings['shape'] = (224, 224, 3)
    settings['nb_classes'] = classes
    # use pre-trained weights otherwise left empty
    settings['weights'] = ""
    # settings['weights'] = "../models_%s/model_%s_%s_weights.h5" % (
    #     config['gpu_unit'], config['name'], optimization['optimizer'])
    settings['verbose'] = 1

    config['settings'] = settings
    config['optimization'] = optimization
    config['training'] = training
    config['testing'] = testing
    params = "epochs_%s_steps_%s_batch_size_%s" % (
        training['epochs'], training['steps_per_epoch'], training['train_batch_size'])

    config['name'] = "%s__%s__%s__%s" % (dataset, name, index, params)

    return config


def create_model(shape, num_classes):
    """Creates appropriate model for the input data."""
    model = Sequential()
    layer1 = Conv2D(96, (11, 11), strides=(4), padding='same', input_shape=shape)

    model.add(layer1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    layer2 = Conv2D(128, (5, 5), padding='same')

    model.add(layer2)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    layer3 = Conv2D(190, (3, 3), padding='same')
    model.add(layer3)
    model.add(Activation('relu'))

    layer4 = Conv2D(190, (3, 3), padding='same')
    model.add(layer4)
    model.add(Activation('relu'))

    layer5 = Conv2D(128, (3, 3), padding='same')
    model.add(layer5)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

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

def get_gpu_configuration():
    """Determines GPU unit configuration. Since this is essential it will exit otherwise.
    """
    gpu_config = input("select GPU type (0 - 'Nvidia 1080', 1 - 'Nvidia Tesla'):")
    if gpu_config == "0":
        return "1080"
    elif gpu_config == "1":
        return "tesla"
    else:
        print("You have to specify GPU unit.\n"
              "Exiting!")
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
        with open("../models_%s/%s__parameters.json" % (
                config['gpu_unit'], config['name']), 'w') as parameters, \
             open("../models_%s/%s__configuration.json" % (
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
