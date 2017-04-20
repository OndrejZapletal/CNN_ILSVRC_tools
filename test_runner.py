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

def load_model():
    pass

def check_for_new_models(list_of_models):
    list_of_files = glob.glob('../models/model*.txt')
    new_models = []
    for model in list_of_files:
        if model not in list_of_models:
            new_models.append(model)
    return new_models

def update_solved_models_count():
    pass

def main_loop():
    list_of_models = []
    while True:
        new_modules = check_for_new_models(list_of_models)
        if new_modules:
            list_of_models += new_modules
            print("Found a new model.")
        wait_time_interval(10)

def wait_time_interval(interval):
    time.sleep(interval)

if __name__ == '__main__':
    main_loop()
