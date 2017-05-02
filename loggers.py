import logging


def create_datasets_logger():
    """Logger for dataset_prepper.py """
    # creation of logger
    logger = logging.getLogger("datasets")
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('../logs/dataset_prepper.log')
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

def create_performance_logger():
    """ Logger for evaluator.py """
    # creation of logger
    logger = logging.getLogger("test_performance")
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('../logs/test_perormance.log')
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

def create_evaluation_logger():
    """ Logger for evaluator.py """
    # creation of logger
    logger = logging.getLogger("evaluator")
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('../logs/evaluation.log')
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

def create_application_logger():
    """Function that creates logger for this module. """
    # creation of logger
    logger = logging.getLogger("application")
    logger.setLevel(logging.DEBUG)

    # logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # file handler
    handler_dbg = logging.FileHandler('../logs/application.log')
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

LOGGER_APP = create_application_logger()

LOGGER_DATASET = create_datasets_logger()

LOGGER_EVALUATION = create_evaluation_logger()

LOGGER_TEST_PERFORMANCE = create_performance_logger()
