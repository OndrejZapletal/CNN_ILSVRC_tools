#!/usr/bin/env python3
"""This is test of keras library."""

import glob
import json
import logging
import re


LAYER_TYPE = ["Conv2D", "MaxPooling2D", "Dense"]
MODEL_PATH = "../trained_models/"
REGEX_MODEL_PATH = MODEL_PATH.replace(".", r"\.")
RESULTS_DESTINATION = "/home/derekin/projects/thesis/document/text/"

def create_logger():
    """Function that creates logger for this module. """
    # creation of logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('../logs/extractor.log')
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


def check_for_new_models():
    """TODO: retrun value of each model_name should contain only unique part of the name.
    i.e time-stamp and iteration.
    """
    list_of_files = sorted(glob.glob(MODEL_PATH + 'model_*_parameters.json'))
    new_models = []
    for file_name in list_of_files:
        model_name = parse_model_name(file_name)
        new_models.append(model_name)
    return new_models


def parse_model_name(file_name):
    """Parser is extracting part of the name betwen words 'model_' and '_parameters.json'."""
    file_parser = re.compile(REGEX_MODEL_PATH + r"model_(.*)_parameters\.json")
    result = file_parser.match(file_name).group(1)
    LOGGER_APP.debug(result)
    if not result:
        LOGGER_APP.error("Name of model wasn't parsed.")
        Exception("Name of model wasn't parsed.")
    # TODO
    return result


def get_model_json(source):
    """TODO: This function will read vales from text file to set model paramters."""
    with open(MODEL_PATH + "model_%s_parameters.json" % source, 'r') as json_file:
        model_json = json_file.read()
        return json.loads(model_json)


def get_model_configuration_from_json(source):
    """TODO: This function will read vales from JSON to configure model. """
    with open(MODEL_PATH + "model_%s_configuration.json" % source, 'r') as json_file:
        return json.loads(json_file.read())

def main_loop():
    """Main program loop."""
    list_of_tables = []
    found_models = check_for_new_models()
    for model_name in found_models:
        model_json = get_model_json(model_name)
        table = analyze_json(model_json, model_name)
        list_of_tables.append(table)

    with open(RESULTS_DESTINATION + "results.org", 'w') as results_file:
        for table in list_of_tables:
            results_file.write(table)
            results_file.write("\n")


def analyze_json(json_data, model_name):
    print("Analyzing model %s" % model_name)
    layers = []
    for layer in json_data['config']:
        layers.append(analyze_layer(layer))
    return create_table(layers, model_name)


def analyze_layer(layer):
    result = []
    name = layer['class_name']
    result.append(name)

    if name == "Conv2D":
        result.append(layer['config']['filters'])
        result.append(layer['config']['kernel_size'])
    elif name == "MaxPooling2D":
        result.append(layer['config']['pool_size'])
    elif name == "Dense":
        result.append(layer['config']['units'])
    elif name == "Activation":
        result.append(layer['config']['activation'])
    elif name == "Dropout":
        result.append(layer['config']['rate'])
    else:
        pass

    return result

def get_header(name):
    tab_name = name
    caption_name = name.replace("_", "\\textunderscore ")
    return "#+NAME: tab:%s\n" \
        "#+CAPTION: Structure of model %s\n" \
        "#+ATTR_LATEX: :align |l|l|c|c|c|c|\n" \
        "|-|\n |layer|name|kernels|size of kernel|" \
        "activation function|regularization|\n|-|\n" % (tab_name, caption_name)

def get_middle():
    return "|-|\n|layer|name|neurons|-|activation function|regularization|\n|-"

def get_footer():
    return "|-|\n"

def create_table(layers, model_name):
    table = get_header(model_name)
    index = 1
    row = ""
    previous_regularized = False
    for layer in layers:
        if layer[0] in LAYER_TYPE:
            if index != 1:
                if not previous_regularized:
                    table += "%s | - |\n" % row
                else:
                    table += "%s |\n" % row
                    previous_regularized = False

            if layer[0] == "MaxPooling2D":
                row = "| %s | %s | - | (%s, %s) | -  " % (index, layer[0], layer[1][0], layer[1][1])
            elif layer[0] == 'Conv2D':
                row = "| %s | %s | %s | (%s, %s) " % (index, layer[0], layer[1], layer[2][0], layer[2][1])
            elif layer[0] == "Dense":
                row = "| %s | %s | %s | - " % (index, layer[0], layer[1])
            index += 1

        elif layer[0] == 'Flatten':
            table += row + "|\n" + get_middle()
            row = ""
        elif layer[0] == 'Dropout':
            row += '| Dropout(%s)' % (layer[1])
            previous_regularized = True
        elif layer[0] == 'Activation':
            row += '| %s ' % (layer[1])
    table += get_footer()


    return table

if __name__ == '__main__':
    main_loop()
