#!/usr/bin/env python3

"""Module used to evaluate created modules"""

import os
import sys
import time
from datetime import datetime

from keras.callbacks import CSVLogger
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import np_utils

from dataset_prepper import generate_test_data, generate_train_data
from loggers import LOGGER_EVALUATION, LOGGER_TEST_PERFORMANCE

TEMP_FILE = '#in_progress#'

def create_temp(model_data, optimizer):
    """Create temp file with basic model settings."""
    with open(TEMP_FILE, 'w') as temp_file:
        temp_file.write("%s\n" % datetime.utcnow())
        temp_file.write("%s\n" % str(optimizer))

        for item in model_data[1].values():
            temp_file.write("%s\n" % str(item))


def remove_temp():
    """Remove temp file after training of model is finished."""
    try:
        os.remove(TEMP_FILE)
    except OSError:
        LOGGER_EVALUATION.debug("This might be a problem since this shuldn't happen!")


def evaluate(model_data):
    """ Function will test results of different optimizers. """
    model = model_data[0]
    config = model_data[1]

    for optimizer in config['optimization']['optimizers']:
        execution_start = time.time()
        try:
            LOGGER_EVALUATION.info('Testing model %s with optimizer %s',
                                   str(config['name']), optimizer['name'])
            create_temp(model_data, optimizer['name'])
            if config['settings']['source']:
                trained_model = fit_generator_model(model, config, optimizer)
            else:
                # trained_model = fit_model(model_data)
                pass
            save_trained_model(trained_model, config, optimizer, execution_start)
            remove_temp()
        except Exception as general_exception:
            LOGGER_EVALUATION.info("Unexpected Error during model evaluation:\n" \
                            "Try to delete model that is causing the problem")
            LOGGER_EVALUATION.info(general_exception)
            remove_temp()
            return False
    return True


def save_trained_model(model, config, optimizer, start):
    """Save model to JSON and weights to h5py."""
    directory = "../trained_models_%s" % config['gpu_unit']

    try:
        with open("%s/model_%s_parameters.json" % (directory, config['name']),
                  'w') as model_paramters:
            model_paramters.write(model.to_json())

        with open("%s/model_%s_statistics.log" % (directory, config['name']),
                  'w') as model_statistics:

            model_statistics.write("start: %s\n"
                                   "end: %s\n"
                                   "difference: %s" % (
                                       start, time.time(), time.time() - start))

        model.save_weights("%s/model_%s_%s_weights.h5" % (
            directory, config['name'], optimizer['name']))

        LOGGER_EVALUATION.debug("trained model %s with optimizer %s saved",
                                config['name'],
                                optimizer['name'])
        return True
    except Exception as general_exception:
        LOGGER_EVALUATION.error(general_exception, sys.exc_info())
        return False


def get_optimizer_data(optimizer):
    """ Return optimizer data. """
    if optimizer['data']:
        if optimizer['name'] == 'sgd':
            opt_ref = SGD()
            return opt_ref.from_config(optimizer['data'])
        elif optimizer['name'] == 'adam':
            opt_ref = Adam()
            return opt_ref.from_config(optimizer['data'])
        elif optimizer['name'] == 'nadam':
            opt_ref = Nadam()
            return opt_ref.from_config(optimizer['data'])
    else:
        return optimizer['name']


def fit_generator_model(model, config, optimizer):
    """Trains Neural Network with available data."""

    csv_logger = CSVLogger(
        '../trained_models_%s/model_%s_%s_performance.log' % (
            config['gpu_unit'], config['name'], optimizer['name']))

    if config['settings']['weights']:
        if os.path.isfile(config['settings']['weights']):
            try:
                LOGGER_EVALUATION.info("Loading weights for %s", config['name'])
                model.load_weights(config['settings']['weights'])
            except Exception as general_exception:
                LOGGER_EVALUATION.info(general_exception)

    model.compile(
        loss=config['optimization']['loss'],
        optimizer=get_optimizer_data(optimizer),
        metrics=config['optimization']['metrics'],)

    model.fit_generator(
        generator=generate_train_data(config['settings']['source'],
                                      config['training']['train_batch_size']),
        steps_per_epoch=config['training']['steps_per_epoch'],
        epochs=config['training']['epochs'],
        verbose=config['settings']['verbose'],
        callbacks=[csv_logger],
        validation_data=generate_test_data(config['settings']['source'],
                                           config['testing']['test_batch_size']),
        validation_steps=config['testing']['validation_steps'])
    return model


# def load_cifar_dataset():
#     """ Load data from CIFAR10 database."""
#     nb_classes = 10
#     max_val = 255
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= max_val
#     x_test /= max_val
#     y_train = np_utils.to_categorical(y_train, nb_classes)
#     y_test = np_utils.to_categorical(y_test, nb_classes)
#     return (x_train, y_train), (x_test, y_test)


# def fit_model(model_data, optimizer):
#     """Trains Neural Network with available data."""
#     model = model_data[0]
#     config = model_data[1]
#     csv_logger = CSVLogger(
#         '../trained_models/model_%s_%s_performance.log' % (config['name'], optimizer))
#     train_data, test_data = load_cifar_dataset()
#     model.compile(
#         loss=config['loss'],
#         optimizer=optimizer,
#         metrics=[config['metrics']])
#     model.fit(
#         train_data[0],
#         train_data[1],
#         batch_size=config['batch_size'],
#         epochs=config['epochs'],
#         verbose=config['verbose'],
#         validation_data=test_data,
#         callbacks=[csv_logger])
#     # evaluate_net(model_data, test_data, i)
#     return model


# def evaluate_net(model_data, test_data, epoch):
#     """Evaluates performance of the network on the test data."""
#     model = model_data[0]
#     config = model_data[1]
#     score = model.evaluate(test_data[0], test_data[1], verbose=0)

#     predicted_classes = model.predict(test_data[0])
#     predicted_classes_indexes = [np.argmax(item) for item in predicted_classes]
#     test_data_indexes = [np.argmax(item) for item in test_data[1]]

#     LOGGER_TEST_PERFORMANCE.info(
#         'model: %s, epoch: %s, accuracy: %s ', config['name'], epoch, score[1])
#     correct_indexes = [
#         i for i, _ in enumerate(test_data_indexes)
#         if test_data_indexes[i] == predicted_classes_indexes[i]
#     ]
#     incorrect_indexes = [
#         i for i, _ in enumerate(test_data_indexes)
#         if test_data_indexes[i] != predicted_classes_indexes[i]
#     ]

#     LOGGER_EVALUATION.debug("Correctly guessed in epoch %s: %s", epoch, len(correct_indexes))
#     LOGGER_EVALUATION.debug("Incorrectly guessed in epoch %s: %s", epoch, len(incorrect_indexes))

#     return score
