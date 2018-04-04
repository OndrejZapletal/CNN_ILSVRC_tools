#!/usr/bin/env python3

"""Module used to evaluate created modules"""

import os
import sys
import time
from datetime import datetime

from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import SGD, Adam, Nadam

from dataset_prepper import generate_test_data, generate_train_data
from loggers import LOGGER_EVALUATION

TEMP_FILE = 'in_progress'

def create_temp(config, optimizer):
	"""Create temp file with basic model settings."""
	with open("#%s#%s#" % (TEMP_FILE, config['gpu_unit']), 'w') as temp_file:
		temp_file.write("%s\n" % datetime.utcnow())
		temp_file.write("%s\n" % str(optimizer))

		for item in config.values():
			temp_file.write("%s\n" % str(item))


def remove_temp(config):
	"""Remove temp file after training of model is finished."""
	try:
		os.remove("#%s#%s#" % (TEMP_FILE, config['gpu_unit']))
	except OSError:
		LOGGER_EVALUATION.debug("This might be a problem since this shuldn't happen!")


def evaluate(model_data):
	""" Function will test results of different optimizers. """
	model = model_data[0]
	config = model_data[1]

	for optimizer in config['optimization']['optimizers']:
		execution_start = time.time()
		try:
			LOGGER_EVALUATION.info(
				'Testing model %s with optimizer %s',
				str(config['name']), optimizer['name']
			)
			create_temp(config, optimizer['name'])
			trained_model = fit_generator_model(model, config, optimizer)
			save_trained_model(trained_model, config, optimizer, execution_start)
			remove_temp(config)
		except Exception as general_exception:
			LOGGER_EVALUATION.info(
				"Unexpected Error during model evaluation:\n" \
				"Try to delete model that is causing the problem"
			)
			LOGGER_EVALUATION.info(general_exception)
			remove_temp(config)
			return False
	return True


def save_trained_model(model, config, optimizer, start):
	"""Save model to JSON and weights to h5py."""
	directory = "../trained_models"

	try:
		with open("%s/%s__parameters.json" % (directory, config['name']), 'w') as model_paramters:
			model_paramters.write(model.to_json())

		with open("%s/%s__%s__statistics.log" % (directory, config['name'], optimizer['name']), 'w') as model_statistics:

			model_statistics.write("start: %s\n"
								   "end: %s\n"
								   "difference: %s" % (
									   start, time.time(), time.time() - start))

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
	directory = "../trained_models"
	csv_logger = CSVLogger(
		'%s/%s__%s__performance.log' % (
			directory, config['name'], optimizer['name']))
	checkpointer = ModelCheckpoint(
		'%s/%s__%s__weights_epoch_{epoch:02d}_val_acc_{val_acc:.2f}.hdf5' % (
			directory, config['name'], optimizer['name']),
		monitor='val_acc',
		verbose=1,
		save_weights_only=True)

	if config['settings']['weights']:
		if os.path.isfile(config['settings']['weights']):
			try:
				LOGGER_EVALUATION.info("Loading weights for %s", config['name'])
				model.load_weights(config['settings']['weights'])
			except Exception as general_exception:
				LOGGER_EVALUATION.info(general_exception)

	model.compile(
		loss = config['optimization']['loss'],
		optimizer = get_optimizer_data(optimizer),
		metrics = config['optimization']['metrics']
	)

	model.fit_generator(
		generator = generate_train_data(
			config['settings']['source'],
			config['training']['train_batch_size']
		),
		steps_per_epoch = config['training']['steps_per_epoch'],
		epochs = config['training']['epochs'],
		verbose = config['settings']['verbose'],
		callbacks = [csv_logger, checkpointer],
		validation_data = generate_test_data(
			config['settings']['source'],
			config['testing']['test_batch_size']
		),
		validation_steps = config['testing']['validation_steps'],
		max_q_size = 10,
		workers = 4,
		pickle_safe = True
	)
	return model
