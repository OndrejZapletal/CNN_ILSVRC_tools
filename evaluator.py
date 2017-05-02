#!/usr/bin/env python3

"""Module used to evaluate created modules"""

import os
import sys
from datetime import datetime

import numpy as np
from keras.callbacks import CSVLogger
from keras.datasets import cifar10
from keras.utils import np_utils

from dataset_prepper import generate_data
from loggers import LOGGER_EVALUATION, LOGGER_TEST_PERFORMANCE

TEMP_FILE = '#in_progress#'


def create_temp(model_data, optimizers):
    """Create temp file with basic model settings."""
    with open(TEMP_FILE, 'w') as temp_file:
        temp_file.write("%s\n" % datetime.utcnow())
        temp_file.write("%s\n" % str(model_data[1]['name']))
        temp_file.write("%s\n" % str(optimizers))
        temp_file.write("%s\n" % str(model_data[1]['batch_size']))
        temp_file.write("%s\n" % str(model_data[1]['epochs']))
        temp_file.write("%s" % str(model_data[1]['verbose']))


def remove_temp():
    """Remove temp file after training of model is finished."""
    try:
        os.remove(TEMP_FILE)
    except OSError:
        LOGGER_EVALUATION.debug("This might be a problem since this shuldn't happen!")


def evaluate(model_data, optimizers):
    """ Function will test results of different optimizers. """
    for optimizer in optimizers:
        try:
            LOGGER_EVALUATION.info('Testing model %s with optimizer %s', str(model_data[1]['name']), optimizer)
            create_temp(model_data, optimizers)
            if not model_data[1]['source']:
                trained_model = fit_model(model_data, optimizer)
            else:
                trained_model = fit_generator_model(model_data, optimizer)
            save_model_to_file(trained_model, model_data[1]['name'], optimizer)
            remove_temp()
        except Exception as general_exception:
            LOGGER_EVALUATION.info("Unexpected Error during model evaluation:\n" \
                            "Try to delete model that is causing the problem")
            LOGGER_EVALUATION.info(general_exception)
            remove_temp()
    return True


def save_model_to_file(model, model_name, optimizer):
    """Save model to json and weights to h5py."""
    try:
        with open("../trained_models/model_%s_parameters.json" % model_name, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights("../trained_models/model_%s_%s_weights.h5" % (model_name, optimizer))
        LOGGER_EVALUATION.debug("model %s saved", model_name)
        return True
    except Exception as general_exception:
        LOGGER_EVALUATION.error(general_exception, sys.exc_info())
        return False


def evaluate_net(model_data, test_data, epoch):
    """Evaluates performance of the network on the test data."""
    model = model_data[0]
    model_configuration = model_data[1]
    score = model.evaluate(test_data[0], test_data[1], verbose=0)

    predicted_classes = model.predict(test_data[0])
    predicted_classes_indexes = [np.argmax(item) for item in predicted_classes]
    test_data_indexes = [np.argmax(item) for item in test_data[1]]

    LOGGER_TEST_PERFORMANCE.info(
        'model: %s, epoch: %s, accuracy: %s ', model_configuration['name'], epoch, score[1])
    correct_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] == predicted_classes_indexes[i]
    ]
    incorrect_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] != predicted_classes_indexes[i]
    ]

    LOGGER_EVALUATION.debug("Correctly guessed in epoch %s: %s", epoch, len(correct_indexes))
    LOGGER_EVALUATION.debug("Incorrectly guessed in epoch %s: %s", epoch, len(incorrect_indexes))

    return score

def fit_generator_model(model_data, optimizer):
    """Trains Neural Network with available data."""
    model = model_data[0]
    model_configuration = model_data[1]
    csv_logger = CSVLogger(
        '../trained_models/model_%s_%s_performance.log' % (model_configuration['name'], optimizer))
    model.compile(
        loss=model_configuration['loss'],
        optimizer=optimizer,
        metrics=[model_configuration['metrics']])
    model.fit_generator(
        generator=generate_data(model_configuration['source'], "train", model_configuration['train_batch_size']),
        steps_per_epoch=model_configuration['steps_per_epoch'],
        epochs=model_configuration['epochs'],
        verbose=model_configuration['verbose'],
        callbacks=[csv_logger],
        validation_data=generate_data(model_configuration['source'], "test", model_configuration['test_batch_size']),
        validation_steps=model_configuration['validation_steps'])


    # evaluate_net(model_data, test_data, i)
    return model


def fit_model(model_data, optimizer):
    """Trains Neural Network with available data."""
    model = model_data[0]
    model_configuration = model_data[1]
    csv_logger = CSVLogger(
        '../trained_models/model_%s_%s_performance.log' % (model_configuration['name'], optimizer))
    train_data, test_data = load_cifar_dataset()
    model.compile(
        loss=model_configuration['loss'],
        optimizer=optimizer,
        metrics=[model_configuration['metrics']])
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=model_configuration['batch_size'],
        epochs=model_configuration['epochs'],
        verbose=model_configuration['verbose'],
        validation_data=test_data,
        callbacks=[csv_logger])
    # evaluate_net(model_data, test_data, i)
    return model


def load_cifar_dataset():
    """TODO:
    Think how to make this as independet from the rest of the script as possible.
    Load data from CIFAR10 database."""
    nb_classes = 10
    max_val = 255
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= max_val
    x_test /= max_val
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return (x_train, y_train), (x_test, y_test)
