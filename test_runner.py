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
import time


# from keras_module import test_model


def load_model(source_name):
    """TODO: Make sure that you are getting correct information from both configuration and parameter files."""

    model = get_model_parameters(source_name), get_model_configuration(source_name)
    return model


def check_for_new_models(list_of_models):
    """TODO: retrun value of each model_name should contain only unique part of the name.
    i.e timestamp and itteration.
    """
    list_of_files = glob.glob('../models/model*.txt')
    new_models = []
    for model_name in list_of_files:
        if model_name not in list_of_models:
            new_models.append(model_name)
    return new_models


def get_model_configuration(source):
    """TODO: This function will read vales from JSON to configure model.
    """
    return source


def get_model_parameters(source):
    """TODO: This function will read vales from text file to set model paramters."""
    return source


def load_dataset():
    """TODO:
    Think how to make this as independet from the rest of the script as possible."""

    return 1


def main_loop():
    """Main program loop.
    TODO: So far it look promising but it needs to be checked in order to determine if it does what it should.
    """
    trained_models = []
    dataset = load_dataset()
    optimizers = ['adam', 'nadam']
    while True:
        new_models = check_for_new_models(trained_models)
        if new_models:
            for model_name in new_models:
                model_descrition = load_model(model_name)
                model_sucessfully_tested = False
                try:
                    model_sucessfully_tested = test_model(model_descrition, optimizers, dataset)
                except:
                    # General exception, TODO: find better exception
                    print("Model %s"% model_descrition)

                if model_sucessfully_tested:
                    trained_models.append(model_name)
                    print("Found model: %s" % model_name)
        wait_time_interval(10)


def test_model(model_descrition, optimizers, dataset):
    """TEST FUNCTION"""
    return True


def wait_time_interval(interval):
    time.sleep(interval)


if __name__ == '__main__':
    main_loop()
