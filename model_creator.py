#!/usr/bin/env python3
"""This is test of keras library."""

import json
import sys

from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Nadam


def configuration_imagenet(index, params, gpu, name):
    """Create JSON file with model configuration. """

    name = "%s_epochs_%s_batch_size_%s" % (name, params[0], params[1])

    dataset = "image_net_40_cat"
    classes = 40

    config, settings, optimization, training, testing = {}, {}, {}, {}, {}
    config['gpu_unit'] = gpu
    optimization['loss'] = "categorical_crossentropy"
    optimization['metrics'] = ["accuracy"]
    optimization['optimizers'] = [get_optimizer(Adam(0.00008, 0.8, 0.999, 0.0))]
    training['epochs'] = params[0]
    training['steps_per_epoch'] = params[1]
    training['train_batch_size'] = 20

    testing['validation_steps'] = 182
    testing['test_batch_size'] = 10

    # Source for Dataset. If empty the CIFAR10 database is used
    settings['source'] = "../datasets/%s.h5" % dataset
    settings['shape'] = (224, 224, 3)
    settings['nb_classes'] = classes
    settings['weights'] = ""
    settings['verbose'] = 1

    config['settings'] = settings
    config['optimization'] = optimization
    config['training'] = training
    config['testing'] = testing

    params = "epochs_%s_steps_per_epoch_%s_batch_size_%s" % (
        training['epochs'], training['steps_per_epoch'], training['train_batch_size'])


    config['name'] = "%s__%s__%s__%s" % (dataset, name, index, params)

    return config


def create_model_imagenet1(shape, num_classes):
    """Creates appropriate model for the input data."""
    model = Sequential()
    layer1 = Conv2D(48, (11, 11), strides=(4), padding='same', input_shape=shape)

    model.add(layer1)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    layer2 = Conv2D(64, (5, 5), padding='same')
    model.add(layer2)
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    layer4 = Conv2D(95, (3, 3), padding='same')
    model.add(layer4)
    model.add(Activation('relu'))

    layer5 = Conv2D(64, (3, 3), padding='same')
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


def create_model_mnist(shape, num_classes):
    """Creates appropriate model for the input data."""
    model = Sequential()
    layer1 = Conv2D(32, (3, 3), padding='same', input_shape=shape)

    model.add(layer1)
    model.add(Activation('relu'))


    layer2 = Conv2D(32, (3, 3), padding='same')

    model.add(layer2)
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))

    layer3 = Conv2D(32, (3, 3), padding='same')
    model.add(layer3)
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def create_model_cifar10(shape, num_classes):
    """Creates appropriate model for the input data."""
    model = Sequential()
    layer1 = Conv2D(32, (3, 3), padding='same', input_shape=shape)

    model.add(layer1)
    model.add(Activation('relu'))


    layer2 = Conv2D(32, (3, 3), padding='same')

    model.add(layer2)
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2)))

    layer3 = Conv2D(32, (3, 3), padding='same')
    model.add(layer3)
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def configuration_mnist(index):
    """Create JSON file with model configuration. """
    name = "testing_simplest_model"
    dataset = "mnist"
    classes = 10

    config, settings, optimization, training, testing = {}, {}, {}, {}, {}
    config['gpu_unit'] = get_gpu_configuration()


    optimization['loss'] = "categorical_crossentropy"
    optimization['metrics'] = ["accuracy"]
    optimization['optimizers'] = [get_optimizer("adam"),
                                  get_optimizer("nadam"),
                                  get_optimizer("sgd")]
    training['epochs'] = 30
    training['train_batch_size'] = 500

    # Source for Dataset. If empty the CIFAR10 database is used
    settings['source'] = dataset
    settings['shape'] = (28, 28, 1)
    settings['nb_classes'] = classes
    # use pre-trained weights otherwise left empty
    settings['weights'] = ""
    settings['verbose'] = 1

    config['settings'] = settings
    config['optimization'] = optimization
    config['training'] = training
    config['testing'] = testing

    params = "epochs_%s_batch_size_%s" % (
        training['epochs'], training['train_batch_size'])

    config['name'] = "%s__%s__%s__%s" % (dataset, name, index, params)

    return config


def configuration_cifar10(index):
    """Create JSON file with model configuration. """
    name = "testing_simplest_model"
    dataset = "cifar10"
    classes = 10

    config, settings, optimization, training, testing = {}, {}, {}, {}, {}
    config['gpu_unit'] = get_gpu_configuration()


    optimization['loss'] = "categorical_crossentropy"
    optimization['metrics'] = ["accuracy"]
    optimization['optimizers'] = [get_optimizer("adam"),
                                  get_optimizer("nadam"),
                                  get_optimizer("sgd")]
    training['epochs'] = 30
    training['train_batch_size'] = 500

    # Source for Dataset. If empty the CIFAR10 database is used
    settings['source'] = dataset
    settings['shape'] = (32, 32, 3)
    settings['nb_classes'] = classes
    # use pre-trained weights otherwise left empty
    settings['weights'] = ""
    settings['verbose'] = 1

    config['settings'] = settings
    config['optimization'] = optimization
    config['training'] = training
    config['testing'] = testing

    params = "epochs_%s_batch_size_%s" % (
        training['epochs'], training['train_batch_size'])

    config['name'] = "%s__%s__%s__%s" % (dataset, name, index, params)

    return config


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
        elif isinstance(optimizer, Adam):
            optimizer_ref['name'] = 'adam'
        elif isinstance(optimizer, Nadam):
            optimizer_ref['name'] = 'nadam'

        optimizer_ref['data'] = optimizer.get_config()
    return optimizer_ref


def save_model_to_file(reference, index, params, gpu, name):
    model = reference[0]
    config = reference[1](index, params, gpu, name)
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
    # models = []

    # models.append((create_model_cifar10, configuration_cifar10))
    # models.append((create_model_mnist, configuration_mnist))

    gpu = get_gpu_configuration()
    steps = [500]
    for index, step in enumerate(steps):
        save_model_to_file((create_model_imagenet1, configuration_imagenet),
                           index, (int((1500000)/(step*20)), step), gpu, "4_cl_half_size_1024_fc")

if __name__ == "__main__":
    main()
