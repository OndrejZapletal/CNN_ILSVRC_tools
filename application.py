#!/usr/bin/env python3
"""This is test of keras library."""

import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

from keras.datasets import cifar10
from keras.models import model_from_json
from keras.utils import np_utils

from evaluator import evaluate
from dataset_prepper import load_dataset_from_h5f

from loggers import LOGGER_APP, LOGGER_DATASET, LOGGER_EVALUATION

TEMP_FILE = '#in_progress#'


def load_model_data(source_name):
    """TODO: Make sure that you are getting correct information
    from both configuration and parameter files."""
    model = get_model(source_name)
    configuration = get_model_configuration(source_name)
    return (model, configuration)


def check_for_new_models(list_of_models):
    """TODO: retrun value of each model_name should contain only unique part of the name.
    i.e timestamp and iteration.
    """
    list_of_files = glob.glob('../models/model_*_parameters.json')
    new_models = []
    for file_name in list_of_files:
        try:
            model_name = parse_model_name(file_name)
            if model_name not in list_of_models:
                new_models.append(model_name)
        except IndexError:
            LOGGER_APP.error("Name of model wasn't parsed.")
    return new_models


def parse_model_name(file_name):
    """Parser is extracting part of the name betwen words 'model_' and '_parameters.json'."""
    file_parser = re.compile(r"\.\./models/model_(.*)_parameters\.json")
    result = file_parser.match(file_name).group(1)
    LOGGER_APP.debug(result)
    if not result:
        IndexError("Name of model wasn't parsed.")
    return result


def get_model(source):
    """TODO: This function will read vales from text file to set model paramters."""
    with open("../models/model_%s_parameters.json" % source, 'r') as json_file:
        loaded_model_json = json_file.read()
        return model_from_json(loaded_model_json)


def get_model_configuration(source):
    """TODO: This function will read vales from JSON to configure model. """
    with open("../models/model_%s_configuration.json" % source, 'r') as json_file:
        return json.loads(json_file.read())


def load_dataset(configuration):
    """TODO:
    Think how to make this as independet from the rest of the script as possible.
    Load data from CIFAR10 database."""
    if configuration['source']:
        (x_train, y_train), (x_test, y_test) = load_dataset_from_h5f(configuration['source'])
    else:
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


def load_optimizers():
    """Function loades optimizers from standalone file."""
    optimizers = ['adam', 'nadam']
    with open("../configuration/optimizers.txt", 'r') as optimizers_file:
        optimizers = optimizers_file.read().strip().split(',')
        LOGGER_APP.debug('loading optimizers from file: %s', optimizers)
    return optimizers


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
        LOGGER_APP.debug("This might be a problem since this shuldn't happen!")


def move_model_source(model_name):
    os.rename("../models/model_%s_parameters.json" % model_name,
              "../trained_models/model_%s_parameters.json" % model_name)
    try:
        os.remove("../models/model_%s_configuration.json" % model_name)
    except OSError:
        LOGGER_APP.debug("Problem with removing of source for models configuration.")


def wait_time_interval(interval):
    """Function defines time interval that will be waited between individual test."""
    time.sleep(interval*60)


def main_loop():
    """Main program loop."""
    trained_models = []
    while True:
        try:
            optimizers = load_optimizers()
            new_models = check_for_new_models(trained_models)
            if new_models:
                print("got here")
                for model_name in new_models:
                    model_sucessfully_tested = False
                    LOGGER_APP.info("Found model: %s", str(model_name))
                    try:
                        model_data = load_model_data(model_name)
                        dataset = load_dataset(model_data[1])
                        try:
                            create_temp(model_data, optimizers)
                            model_sucessfully_tested = evaluate(model_data, optimizers, dataset)
                            remove_temp()
                        except Exception as general_exception:
                            LOGGER_APP.info(general_exception, sys.exc_info())
                            LOGGER_APP.info("Unexpected Error: Try to delete model that is causing the problem")
                            remove_temp()
                    except FileNotFoundError:
                        LOGGER_APP.info("Configuration file is missing!")
                    if model_sucessfully_tested:
                        trained_models.append(model_name)
                        LOGGER_APP.info("Successfully tested model: %s", str(model_name))
                        move_model_source(model_name)
            else:
                wait_time_interval(1)
        except Exception as general_exception:
            LOGGER_APP.error(general_exception, sys.exc_info())


if __name__ == '__main__':
    main_loop()
