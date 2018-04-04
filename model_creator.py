#!/usr/bin/env python3
"""This is test of keras library."""

import json
import sys

from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam, Nadam


def configuration_imagenet():
	"""Create JSON file with model configuration. """
	return {
		'optimization': {
			'loss': "categorical_crossentropy",
			'metrics': ["accuracy"],
			'optimizers': [get_optimizer(Adam(0.00008, 0.8, 0.999, 0.0))],
		},
		'training': {
			'epochs': 100,
			'steps_per_epoch': 10,
			'train_batch_size': 20
		},
		'testing': {
			'validation_steps': 182,
			'test_batch_size': 10
		},
		'settings': {
			'source': '../datasets/image_net_10_cat.h5',
			'shape': (224, 224, 3),
			'nb_classes': 10,
			'weights': '',
			'verbose': 1
		},
		'name': 'test_configuration',
		'gpu_unit': '1080'
	}



def create_model_imagenet(shape, num_classes):
	"""Creates appropriate model for the input data."""
	model = Sequential()
	layer1 = Conv2D(48, (11, 11), strides=(4), padding='same', input_shape=shape)

	model.add(layer1)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

	layer2 = Conv2D(64, (5, 5), padding='same')
	model.add(layer2)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

	layer4 = Conv2D(95, (3, 3), padding='same')
	model.add(layer4)
	model.add(Activation('relu'))

	layer5 = Conv2D(64, (3, 3), padding='same')
	model.add(layer5)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	return model



def get_optimizer(optimizer):
	optimizer_ref = {}
	if isinstance(optimizer, str):
		optimizer_ref['name'] = optimizer
		optimizer_ref['data'] = None
	else:
		if isinstance(optimizer, SGD):
			optimizer_ref['name'] = 'sgd'
		elif isinstance(optimizer, Adam):
			optimizer_ref['name'] = 'adam'
		elif isinstance(optimizer, Nadam):
			optimizer_ref['name'] = 'nadam'

		optimizer_ref['data'] = optimizer.get_config()
	return optimizer_ref


def save_model_to_file(model, config):
	try:
		with open("../models/%s__parameters.json" % config['name'], 'w') as parameters_file, \
			 open("../models/%s__configuration.json" % config['name'], 'w') as configuration_file:
			model_specification = model(
				config['settings']['shape'],
				config['settings']['nb_classes']
			)
			parameters_file.write(model_specification.to_json())
			configuration_file.write(json.dumps(config))
			print("model %s saved" % config['name'])
		return True
	except Exception as general_exception:
		print(general_exception, sys.exc_info())
		return False


def main():
	"""main function"""
	save_model_to_file(
		create_model_imagenet,
		configuration_imagenet()
	)

if __name__ == "__main__":
	main()
