#!/usr/bin/env python3
"""This is test of keras library."""

import glob
import json
import logging
import os
import re

LAYER_TYPE = ["Conv2D", "MaxPooling2D", "Dense"]
MODEL_PATH = "../trained_models_1080/"
REGEX_MODEL_PATH = MODEL_PATH.replace(".", r"\.")
RESULTS_DESTINATION = "/home/derekin/Dropbox/document/text/"
COLORS = ['blue', 'red', 'green', 'violet', 'yellow', 'orange', 'brown', 'gray']

def create_logger():
	"""Function that creates logger for this module. """
	# creation of logger
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)

	# logging format
	formatter = logging.Formatter('%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s')

	# file handler
	handler_dbg = logging.FileHandler('../logs_1080/extractor.log')
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
	list_of_files = sorted(glob.glob(MODEL_PATH + '*__parameters.json'))
	new_models = []
	for file_name in list_of_files:
		model_title = parse_model_title(file_name)
		new_models.append(model_title)
	return new_models


def parse_model_title(file_name):
	"""Parser is extracting part of the name betwen words 'model_' and '_parameters.json'."""
	regex = REGEX_MODEL_PATH + r"(.*)__parameters\.json"
	file_parser = re.compile(regex)
	result = file_parser.match(file_name).group(1)
	LOGGER_APP.debug(result)
	if not result:
		LOGGER_APP.error("Name of model wasn't parsed.")
		Exception("Name of model wasn't parsed.")
	return result


def check_for_performances():
	"""TODO: retrun value of each model_name should contain only unique part of the name.
	i.e time-stamp and iteration.
	"""
	list_of_files = sorted(glob.glob(MODEL_PATH + '*__performance.log'))
	new_models = []
	for file_name in list_of_files:
		model_name = parse_model_performance_name(file_name)
		new_models.append(model_name)

	return zip(new_models, list_of_files)


def parse_model_performance_name(file_name):
	"""Parser is extracting part of the name betwen words 'model_' and '_parameters.json'."""
	regex = REGEX_MODEL_PATH + r"(.*)__performance\.log"
	file_parser = re.compile(regex)
	result = file_parser.match(file_name).group(1)
	LOGGER_APP.debug(result)
	if not result:
		LOGGER_APP.error("Name of model wasn't parsed.")
		Exception("Name of model wasn't parsed.")
	return result


def get_model_json(source):
	"""TODO: This function will read vales from text file to set model paramters."""
	with open(MODEL_PATH + "%s__parameters.json" % source, 'r') as json_file:
		model_json = json_file.read()
		return json.loads(model_json)


def get_model_configuration(source):
	"""TODO: This function will read vales from JSON to configure model. """
	with open(MODEL_PATH + "%s__configuration.json" % source, 'r') as json_file:
		return json.loads(json_file.read())


def parse_name_from_title(model_title):
	"""Function tries to parse model name from its convoluted title. """
	try:
		title_parser = re.compile("(.*?)__(.*?)__(.*?)__(.*?)")
		reference = title_parser.match(model_title)
		name = reference.group(2)
		return name

	except Exception:
		return model_title


def parse_name_from_performance_title(model_title):
	"""Function tries to parse model name from its convoluted title. """
	try:
		title_parser = re.compile("(.*?)__(.*?)__(.*?)__(.*?)__(.*)")
		reference = title_parser.match(model_title)
		name = reference.group(2)
		optimizer = reference.group(5)
		return (name, optimizer)

	except Exception:
		return model_title

def create_combined_performance(performance_files):
	LOGGER_APP.info("Creating chart for all models")

	chart_training = create_header("Training accuracy")
	chart_testing = create_header("Testing accuracy")

	for index, (model_title, path) in enumerate(performance_files):

		name = parse_name_from_performance_title(model_title)
		if len(name) == 2:
			name = name[0]

		path = os.path.join(os.getcwd(), path)


		chart_training += add_plot(path, name.replace("_", " "), index, 'epoch', 'acc')
		chart_testing += add_plot(path, name.replace("_", " "), index, 'epoch', 'val_acc')

	chart_training += "\n" + create_footer()
	chart_testing += "\n" + create_footer()

	return chart_training + chart_testing

def main_loop():
	"""Main program loop."""
	list_of_tables = []
	list_of_charts = []
	found_models = check_for_new_models()
	found_performances = check_for_performances()

	for model_title in found_models:
		model_name = parse_name_from_title(model_title)
		model_json = get_model_json(model_title)
		table = analyze_jsons(model_json, model_name)
		list_of_tables.append(table)

	for model_title, path in found_performances:
		chart = create_chart_from_performance(model_title, path)
		list_of_charts.append(chart)

	found_performances = check_for_performances()

	combined_chart = create_combined_performance(found_performances)

	with open(RESULTS_DESTINATION + "results.org", 'w') as results_file:
		for table in list_of_tables:
			results_file.write(table)
			results_file.write("\n")

	with open(RESULTS_DESTINATION + "charts.org", 'w') as results_file:
		for chart in list_of_charts:
			results_file.write(chart)
			results_file.write("\n")

	with open(RESULTS_DESTINATION + "combined_chart.org", 'w') as results_file:
		results_file.write(combined_chart)

# def analyze_jsons(json_data, parameters, model_name):
def analyze_jsons(json_data, model_name):
	LOGGER_APP.info("Analyzing model %s", model_name)
	layers = []
	for layer in json_data['config']:
		layers.append(analyze_layer(layer))
		# return create_table(layers, parameters)
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
	caption_name = name.replace("_", " ")
	return "#+NAME: tab:%s\n" \
		"#+CAPTION: Structure of model %s\n" \
		"#+ATTR_LATEX: :align |l|l|c|c|c|c|\n" \
		"|-|\n |layer|name|kernels|size of kernel|" \
		"activation function|regularization|\n|-|\n" % (tab_name, caption_name)


def get_middle():
	return "|-|\n|layer|name|neurons|-|activation function|regularization|\n|-"


def get_footer():
	return "|-|\n"


# def get_params(params):
#     return "|-|\n|batch size|epochs|-|-|-|\n|-|"

# def create_table(layers, parameters):
def create_table(layers, model_name):
	# table = get_header(parameters['name'])
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
			if not previous_regularized:
				table += "%s| - |\n%s" % (row, get_middle())
			else:
				table += "%s|\n%s" % (row, get_middle())
				previous_regularized = False
				row = ""
		elif layer[0] == 'Dropout':
			row += '| Dropout(%s)' % (layer[1])
			previous_regularized = True
		elif layer[0] == 'Activation':
			row += '| %s' % (layer[1])

	# table += get_params(parameters)
	table += row + "| - |\n"
	table += get_footer()
	return table


def create_chart_from_performance(title, path):
	LOGGER_APP.info("Creating chart for model %s", title)

	name = parse_name_from_performance_title(title)

	path = os.path.join(os.getcwd(), path)

	if len(name) == 2:
		chart_text = create_header("%s (%s)" % (name[0].replace("_", " "), name[1]))
	else:
		chart_text = create_header(name.replace("_", " "))

	chart_text += "\n" + add_plot(path, "Train error", 0, 'epoch', 'acc')
	chart_text += "\n" + add_plot(path, "Test error", 1, 'epoch', 'val_acc')
	chart_text += "\n" + create_footer()
	return chart_text


def create_footer():
	"""Create footer data """
	return r"""
	\end{axis}
\end{tikzpicture}
	"""


def create_header(title):
	"""Create header data """
	return r"""
\begin{tikzpicture}
	\begin{axis}[
		title={%s},
		xlabel={epoch},
		ylabel={accuracy [p]},
		ymin=0.0, ymax=1,
		legend pos=south east,
		ymajorgrids=true,
		xmajorgrids=true,
		grid style=dashed,
		scale=1.5,
	]
	""" % title


def add_plot(path, legend, index, x_col, y_col):
	"""Create plot data """
	try:
		color = COLORS[index]
	except IndexError:
		color = "black"

	return r"""
	\addplot[color=%s]
		table [x=%s, y=%s, col sep=comma]
		{%s};
		\addlegendentry{%s}
	""" % (color,
		   x_col,
		   y_col,
		   path,
		   legend)


if __name__ == '__main__':
	main_loop()
