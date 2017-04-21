import logging
import sys

import numpy as np


# from keras.datasets import cifar10
# from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
#                           MaxPooling2D)
# from keras.models import Sequential
# from keras.utils import np_utils


def create_logger():
    # creation of logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('keras_debug.log')
    handler_dbg.setLevel(logging.DEBUG)
    handler_dbg.setFormatter(formatter)

    # stream handler
    handler_inf = logging.StreamHandler()
    handler_inf.setLevel(logging.INFO)
    handler_inf.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler_dbg)
    logger.addHandler(handler_inf)
    return logger


LOGGER_KERAS = create_logger()


def test_model(model, optimizers, dataset):
    """main function"""
    LOGGER_KERAS.info("testing model: %s", str(model))
    find_performance_of_model(model, optimizers, dataset)
    return True

def find_performance_of_model(model_data, optimizers, data):
    """ Function will test results of different optimizers. """
    for optimizer in optimizers:
        LOGGER_KERAS.info('Testing model %s with optimizer %s', str(model_data[1]['name']), optimizer)
        trained_model = train_model(model_data, data[0], optimizer)
        # save_model_to_file(trained_model, '%s_%s' % (i, optimizer))
        evaluate_net(trained_model, data[1])


# def save_model_to_file(model, name):
#     """Save model to json and weigths to h5py."""
#     try:
#         with open("../trained_models/model_%s_architecture.json" % name, 'w') as json_file:
#             json_file.write(model.to_json())
#         # model.save_weights("model_%s_weights.h5" % name)
#         # print("model_%s saved" % name)
#         return True
#     except:
#         print(sys.exc_info())
#         return False


def evaluate_net(model, test_data):
    """Evaluates preformamce of the network on the test data."""
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    LOGGER_KERAS.info('Test score: %s', score[0])
    LOGGER_KERAS.info('Test accuracy: %s', score[1])

    predicted_classes = model.predict(test_data[0])
    predicted_classes_indexes = [np.argmax(item) for item in predicted_classes]
    test_data_indexes = [np.argmax(item) for item in test_data[1]]

    correct_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] == predicted_classes_indexes[i]
    ]
    incorrect_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] != predicted_classes_indexes[i]
    ]

    LOGGER_KERAS.info("Correctly guessed: %s", len(correct_indexes))
    LOGGER_KERAS.info("Incorrectly guessed: %s", len(incorrect_indexes))

    return score

def train_model(model_data, train_data, optimizer):
    """Trains Neural Network with available data."""
    model = model_data[0]
    model_configuration = model_data[1]
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
        validation_split=model_configuration['validation_split'])
    return model
