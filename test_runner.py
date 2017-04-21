#!/usr/bin/env python3
"""This is test of keras library."""

# - folder for models to be done
# - folder for finished models
# - folder for results
# - folder for weights
# - check every 10 minutes
# - find if any new model is present
# - any new file add to queue
# - create temp file signifying that new training started
# - run model and save results and weights

import glob
import json
import logging
import os
import re
import time

from keras.datasets import cifar10
from keras.models import model_from_json
from keras.utils import np_utils

from keras_module import test_model

TEMP_FILE = '#in_progress#'

def create_logger():
    """Function that creates logger for this module. """
    # creation of logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('application_debug.log')
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

LOGGER_APP = create_logger()

def load_model_data(source_name):
    """TODO: Make sure that you are getting correct information from both configuration and parameter files."""
    model = get_model_from_json(source_name)
    configuration = get_model_configuration_from_json(source_name)
    return (model, configuration)


def check_for_new_models(list_of_models):
    """TODO: retrun value of each model_name should contain only unique part of the name.
    i.e timestamp and itteration.
    """
    list_of_files = glob.glob('../models/model_*_parameters.json')
    new_models = []
    for file_name in list_of_files:
        # TODO: Don't forget to handle the exception
        model_name = parse_model_name(file_name)
        if model_name not in list_of_models:
            new_models.append(model_name)
    return new_models

def parse_model_name(file_name):
    """Parser is extracting part of the name betwen words 'model_' and '_parameters.json'."""
    file_parser = re.compile(r"\.\./models/model_(.*)_parameters\.json")
    result = file_parser.match(file_name).group(1)
    LOGGER_APP.debug(result)
    if not result:
        LOGGER_APP.error("Name of model wasn't parsed.")

        Exception("Name of model wasn't parsed.")
    # TODO
    return result


def get_model_from_json(source):
    """TODO: This function will read vales from text file to set model paramters."""
    with open("../models/model_%s_parameters.json" % source, 'r') as json_file:
        loaded_model_json = json_file.read()
        return model_from_json(loaded_model_json)


def get_model_configuration_from_json(source):
    """TODO: This function will read vales from JSON to configure model. """
    with open("../models/model_%s_configuration.json" % source, 'r') as json_file:
        return json.loads(json_file.read())



def load_dataset():
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
    LOGGER_APP.debug("Training matrix shape %s", x_train.shape)
    LOGGER_APP.debug("Testing matrix shape %s", x_test.shape)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return ((x_train, y_train), (x_test, y_test))


def load_optimizers():
    """Function loades optimizers from standalone file."""
    optimizers = ['adam', 'nadam']
    with open("../configuration/optimizers.txt", 'r') as optimizers_file:
        optimizers = optimizers_file.read().strip().split(',')
        LOGGER_APP.debug('loading optimizers from file: %s', optimizers)
    return optimizers


def main_loop():
    """Main program loop.
    TODO: So far it look promising but it needs to be checked in order to determine if it does what it should.
    """
    trained_models = []
    dataset = load_dataset()
    while True:
        optimizers = load_optimizers()
        new_models = check_for_new_models(trained_models)
        if new_models:
            for model_name in new_models:
                model_sucessfully_tested = False

                LOGGER_APP.info("Found model: %s", str(model_name))
                try:

                    model_data = load_model_data(model_name)

                    try:
                        with open(TEMP_FILE, 'w') as temp_file:
                            temp_file.write("%s\n" % time.time())
                            temp_file.write("%s\n" % str(model_data[1]['name']))
                            temp_file.write("%s" % str(optimizers))

                        model_sucessfully_tested = test_model(model_data, optimizers, dataset)

                        try:
                            os.remove(TEMP_FILE)
                        except OSError:
                            LOGGER_APP.debug("This might be a problem since this shuldn't happen!")
                    except:
                        # General exception, TODO: find better exception
                        LOGGER_APP.info("Model: %s", str(model_data[1]['name']))
                        try:
                            os.remove(TEMP_FILE)
                        except OSError:
                            LOGGER_APP.debug("Temp file not present while handeling exception.")
                except FileNotFoundError:
                    LOGGER_APP.info("Configuration file is missing!")

                if model_sucessfully_tested:
                    trained_models.append(model_name)
                    LOGGER_APP.info("Succesfully tested model: %s", str(model_name))
        wait_time_interval(10)



def wait_time_interval(interval):
    """Function defines time interval that will be waited between individual test."""
    time.sleep(interval*60)


if __name__ == '__main__':
    main_loop()
