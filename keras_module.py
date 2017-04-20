import sys

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.utils import np_utils


def test_model(model, optimizers, dataset):
    """main function"""

    loaded_model = load_model_from_file(model)

    find_performance_of_model(loaded_model, optimizers, data)

def find_performance_of_model(models, optimizers, data):
    """ Function will test results of different optimizers.
    """
    for i, model in enumerate(models, start=4):
        for optimizer in optimizers:
            print('\nTesting model %s with optimizer %s' % (i, optimizer))
            trained_model = train_model(model(), data[0], optimizer)
            save_model_to_file(trained_model, '%s_%s' % (i, optimizer))
            evaluate_net(trained_model, data[1])
            print("Layer 0:")
            print(trained_model.layers[0].input_shape)
            print(trained_model.layers[0].output_shape)
            print(trained_model.layers[0].weights)

            print("Layer 1:")
            print(trained_model.layers[1].input_shape)
            print(trained_model.layers[1].output_shape)
            print(trained_model.layers[1].weights)

            print("Layer 2:")
            print(trained_model.layers[2].input_shape)
            print(trained_model.layers[2].output_shape)
            print(trained_model.layers[2].weights)

def load_model_from_file(model):
    """Save model to json and weigths to h5py."""
    try:
        with open("model_%s_architecture.json" % name, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights("model_%s_weights.h5" % name)
        print("model_%s saved" % name)
        return True
    except:
        print(sys.exc_info())
        return False

def save_model_to_file(model, name):
    """Save model to json and weigths to h5py."""
    try:
        with open("model_%s_architecture.json" % name, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights("model_%s_weights.h5" % name)
        print("model_%s saved" % name)
        return True
    except:
        print(sys.exc_info())
        return False

def evaluate_net(model, test_data):
    """Evaluates preformamce of the network on the test data."""
    score = model.evaluate(test_data[0], test_data[1], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

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

    print("Correctly guessed: %s" % len(correct_indexes))
    print("Incorrectly guessed: %s" % len(incorrect_indexes))

    if plot:
        plot_mistakes(incorrect_indexes, test_data, predicted_classes_indexes,
                      test_data_indexes)
    return score

def train_model(model, train_data, optimizer):
    """Trains Neural Network with available data."""
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=128,
        epochs=1,
        verbose=1,
        validation_split=0.1)
    return model
