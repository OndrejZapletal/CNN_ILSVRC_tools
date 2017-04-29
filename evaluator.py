from loggers import LOGGER_EVALUATION, LOGGER_TEST_PERFORMANCE
import sys
import numpy as np
from keras.callbacks import CSVLogger


def evaluate(model_data, optimizers, data):
    """ Function will test results of different optimizers. """
    for optimizer in optimizers:
        LOGGER_EVALUATION.info('Testing model %s with optimizer %s',
                          str(model_data[1]['name']), optimizer)
        trained_model = train_model(model_data, data, optimizer)
        save_model_to_file(trained_model, model_data[1]['name'], optimizer)
    return True


def save_model_to_file(model, model_name, optimizer):
    """Save model to json and weigths to h5py."""
    try:
        with open("../trained_models/model_%s_parameters.json" % model_name, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights("../trained_models/model_%s_%s_weights.h5" % (model_name, optimizer))
        LOGGER_EVALUATION.debug("model %s saved", model_name)
        return True
    except Exception as general_exception:
        LOGGER_EVALUATION.error(general_exception, sys.exc_info())
        return False


def evaluate_net(model_data, test_data, epoch):
    """Evaluates performance of the network on the test data."""
    model = model_data[0]
    model_configuration = model_data[1]
    score = model.evaluate(test_data[0], test_data[1], verbose=0)

    predicted_classes = model.predict(test_data[0])
    predicted_classes_indexes = [np.argmax(item) for item in predicted_classes]
    test_data_indexes = [np.argmax(item) for item in test_data[1]]

    LOGGER_TEST_PERFORMANCE.info(
        'model: %s, epoch: %s, accuracy: %s ', model_configuration['name'], epoch, score[1])
    correct_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] == predicted_classes_indexes[i]
    ]
    incorrect_indexes = [
        i for i, _ in enumerate(test_data_indexes)
        if test_data_indexes[i] != predicted_classes_indexes[i]
    ]

    LOGGER_EVALUATION.debug("Correctly guessed in epoch %s: %s", epoch, len(correct_indexes))
    LOGGER_EVALUATION.debug("Incorrectly guessed in epoch %s: %s", epoch, len(incorrect_indexes))

    return score

def train_model(model_data, data, optimizer):
    """Trains Neural Network with available data."""
    model = model_data[0]
    model_configuration = model_data[1]
    csv_logger = CSVLogger(
        '../trained_models/model_%s_%s_performance.log' % (model_configuration['name'], optimizer))
    train_data = data[0]
    test_data = data[1]
    model.compile(
        loss=model_configuration['loss'],
        optimizer=optimizer,
        metrics=[model_configuration['metrics']])
    model.fit(
        train_data[0],
        train_data[1],
        batch_size=model_configuration['batch_size'],
        epochs=model_configuration['epochs'],
        verbose=model_configuration['verbose'],
        validation_data=test_data,
        # validation_split=model_configuration['validation_split'],
        callbacks=[csv_logger])
    # evaluate_net(model_data, test_data, i)
    return model
