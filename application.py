#!/usr/bin/env python3
"""This is test of keras library."""

import glob
import json
import os
import re
import time

from keras.models import model_from_json

from evaluator import evaluate
from loggers import LOGGER_APP, LOGGER_DATASET, LOGGER_EVALUATION


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
	list_of_files = glob.glob('../models/*__parameters.json')
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
	file_parser = re.compile(r"\.\./models/(.*)__parameters\.json")
	result = file_parser.match(file_name).group(1)
	LOGGER_APP.debug(result)
	if not result:
		IndexError("Name of model wasn't parsed.")
	return result


def get_model(source):
	"""TODO: This function will read vales from text file to set model parameters."""
	with open("../models/%s__parameters.json" % (source), 'r') as json_file:
		loaded_model_json = json_file.read()
		return model_from_json(loaded_model_json)


def get_model_configuration(source):
	"""TODO: This function will read vales from JSON to configure model. """
	with open("../models/%s__configuration.json" % (source), 'r') as json_file:
		return json.loads(json_file.read())



def move_model_source(model_name):
	""" When the evaluation of the model is finished both json files are
	moved to '../trained_models/' folder
	"""
	os.rename(
		"../models/%s__parameters.json" % (model_name),
		"../trained_models/%s__parameters.json" % (model_name)
	)
	os.rename(
		"../models/%s__configuration.json" % (model_name),
		"../trained_models/%s__configuration.json" % (model_name)
	)


def wait_time_interval(interval):
	"""Function defines time interval that will be waited between individual test."""
	time.sleep(interval*60)


def main_loop():
	"""Main program loop."""
	trained_models = []
	while True:
		try:
			new_models = check_for_new_models(trained_models)
			for model_name in new_models:
				LOGGER_APP.info("Found model: %s", str(model_name))
				try:
					model_data = load_model_data(model_name)
				except FileNotFoundError:
					LOGGER_APP.info("Configuration file is missing!")
					continue

				if evaluate(model_data):
					trained_models.append(model_name)
					LOGGER_APP.info("Successfully tested model: %s", str(model_name))
					move_model_source(model_name)

		except Exception as general_exception:
			LOGGER_APP.error(general_exception)
			wait_time_interval(1)


if __name__ == '__main__':
	main_loop()
