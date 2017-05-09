#!/usr/bin/env python3
"This project will be used for preprocessing of ImageNet dataset for master thesis"
import glob
import os
import os.path
from random import randint, sample

import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from scipy.misc import imresize
from scipy.ndimage import imread

from loggers import LOGGER_DATASET

# SOURCE = os.path.expanduser("~/projects/ILSVRC/source_dataset/")
SOURCE = "/media/derekin/My Book/Zaloha/Ondra/ILSVRC/Data/DET/train/ILSVRC2013_train/"
DESTINATION = os.path.expanduser("~/projects/ILSVRC/destination_dataset/")

HDF5_CAT = "../datasets/image_net_categorized.h5"
HDF5_SPL = "../datasets/image_net_split.h5"
HDF5_PRP = "../datasets/image_net_40_cat.h5"

TRAIN_SIZE = 224
DIFF = 256 - TRAIN_SIZE

SIZE = 256
MAX_VAL = 255
RESHAPE_STEP = 300
AVR_IMAGES_PER_CLASS = 500
TRAIN_TEST_RATION = 0.9
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 10
BATCH_SIZE = 500
DELIMITER = "*" * 30
NUM_OF_SELECTED_CLASSES = 100


def prepare_dataset_from_hf5(hf5_file_source, hf5_file_destination):
    """ Think how to make this as independent from the rest of the script as possible.
    Load data from CIFAR10 database."""
    LOGGER_DATASET.info(DELIMITER)
    LOGGER_DATASET.info("Preparing dataset")
    LOGGER_DATASET.info(DELIMITER)
    with h5py.File(hf5_file_source, 'r') as hf5_r, h5py.File(hf5_file_destination, 'w') as hf5_w:
        train = hf5_w.create_group("/data/train")
        test = hf5_w.create_group("/data/test")

        nb_classes = hf5_r["data"].attrs['nb_classes']
        train.attrs['batch_size'] = hf5_r["/data/train"].attrs['batch_size']
        test.attrs['batch_size'] = hf5_r["/data/test"].attrs['batch_size']

        x_train_src = hf5_r["/data/train/x"]
        y_train_src = hf5_r["/data/train/y"]
        x_test_src = hf5_r["/data/test/x"]
        y_test_src = hf5_r["/data/test/y"]

        LOGGER_DATASET.info("Loading data from HDF5 file")
        x_train_dst = train.create_dataset("x", x_train_src.shape, dtype=np.float32, chunks=True)
        y_train_dst = train.create_dataset("y", (y_train_src.shape[0], nb_classes), chunks=True)
        x_test_dst = test.create_dataset("x", x_test_src.shape, dtype=np.float32, chunks=True)
        y_test_dst = test.create_dataset("y", (y_test_src.shape[0], nb_classes), chunks=True)

        LOGGER_DATASET.info("Process data by chunks:")

        for lower in range(0, x_train_src.shape[0], BATCH_SIZE):
            upper = lower + BATCH_SIZE \
                    if lower + BATCH_SIZE < x_train_src.shape[0] \
                       else x_train_src.shape[0]
            LOGGER_DATASET.info("processing chunk [%s:%s]", lower, upper)
            x_train_dst[lower:upper] = np.divide(x_train_src[lower:upper], MAX_VAL)
            y_train_dst[lower:upper] = np_utils.to_categorical(y_train_src[lower:upper], nb_classes)

        for lower in range(0, x_test_src.shape[0], BATCH_SIZE):
            upper = lower + BATCH_SIZE \
                    if lower + BATCH_SIZE < x_test_src.shape[0] \
                       else x_test_src.shape[0]
            LOGGER_DATASET.info("processing chunk [%s:%s]", lower, upper)
            x_test_dst[lower:upper] = np.divide(x_test_src[lower:upper], MAX_VAL)
            y_test_dst[lower:upper] = np_utils.to_categorical(y_test_src[lower:upper], nb_classes)

        LOGGER_DATASET.info("finished")


def load_dataset_from_images(source, hdf5_file_name):
    """Function loads images from source directory and loads them into
    numpy ndarrys.
    """
    LOGGER_DATASET.info(DELIMITER)
    LOGGER_DATASET.info("Loading dataset from images:")
    LOGGER_DATASET.info(DELIMITER)
    with h5py.File(hdf5_file_name, 'w') as hf5_w:
        data_grp = hf5_w.create_group("data")
        data_grp.attrs["nb_classes"] = 0
        num_of_images = 0
        directories = sample(sorted(glob.glob(os.path.join(source, "*/"))), NUM_OF_SELECTED_CLASSES)

        for index, dir_path in enumerate(directories, start=1):

            dir_name = os.path.split(os.path.dirname(dir_path))[1]

            LOGGER_DATASET.info("Processing directory %s (%s of %s)",
                                dir_name, index, len(directories))
            dataset = data_grp.create_dataset(
                dir_name, (AVR_IMAGES_PER_CLASS, SIZE, SIZE, 3), maxshape=(None, SIZE, SIZE, 3))

            index = 0
            image_batch = []
            files = sorted(glob.glob(dir_path + "/*.JPEG"))
            for index_f, file_path in enumerate(files, start=1):

                # check whether the size of dataset is big enough for another image
                if dataset.shape[0] <= index:
                    dataset.resize((dataset.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3))

                image_batch.append(get_image(file_path))
                index += 1
                num_of_images += 1

                if len(image_batch) == BATCH_SIZE:
                    dataset[index - BATCH_SIZE:index] = np.stack(image_batch, axis=0)
                    image_batch = []

                LOGGER_DATASET.debug("File %s saved into %s (%s of %s)",
                                     file_path, dataset.name, index_f, len(files))

            if image_batch:
                dataset[index - len(image_batch):index] = np.stack(image_batch, axis=0)

            # at the end of folder resize shape to exactly fit the number of images within dataset
            if dataset.shape[0] > index:
                dataset.resize((index, SIZE, SIZE, 3))

            # at the end of folder increase number of classes
            data_grp.attrs["nb_classes"] += 1
        data_grp.attrs['num_of_images'] = num_of_images


def split_data(hf5_file_name_src, hf5_file_name_dst):
    """Splits data into train and test datesets."""
    LOGGER_DATASET.info(DELIMITER)
    LOGGER_DATASET.info("Splitting date into train and test.")
    LOGGER_DATASET.info(DELIMITER)
    with h5py.File(hf5_file_name_src, 'r') as hf5_r, h5py.File(hf5_file_name_dst, 'w') as hf5_w:
        x_train, y_train, x_test, y_test, data, indexes = prepare_references(hf5_r, hf5_w)
        x_train_batch, y_train_batch, x_test_batch, y_test_batch = [], [], [], []
        index_train, index_test = 0, 0

        num_of_images = hf5_r["data"].attrs['num_of_images']

        for index, (category, image_index) in enumerate(generate_random_indexes(indexes)):
            LOGGER_DATASET.info("splitting image %s of %s (category: %s, image: %s)",
                                 index, num_of_images, category, image_index)
            image = data[category][image_index, :, :, :]

            if index % 10:
                x_train_batch.append(image)
                y_train_batch.append(category)
                index_train += 1
                if len(x_train_batch) == BATCH_SIZE:
                    write_batch(x_train, y_train, x_train_batch,
                                y_train_batch, index_train, "train")
                    x_train_batch, y_train_batch = [], []
            else:
                x_test_batch.append(image)
                y_test_batch.append(category)
                index_test += 1
                if len(x_test_batch) == BATCH_SIZE:
                    write_batch(x_test, y_test, x_test_batch, y_test_batch, index_test, "test")
                    x_test_batch, y_test_batch = [], []

        if x_train_batch:
            write_batch(x_train, y_train, x_train_batch, y_train_batch, index_train, "train")

        if x_test_batch:
            write_batch(x_test, y_test, x_test_batch, y_test_batch, index_test, "test")

        if x_train.shape[0] > index_train:
            LOGGER_DATASET.debug("Decreasing the size of dataset to fit the data exactly.")
            LOGGER_DATASET.info("new x_train shape: %s, original shape: %s",
                                (index_train, SIZE, SIZE, 3), x_train.shape)
            x_train.resize((index_train, SIZE, SIZE, 3))
            LOGGER_DATASET.info("new y_train shape: %s, original shape: %s",
                                (index_train, 1), y_train.shape)
            y_train.resize((index_train, 1))

        if x_test.shape[0] > index_test:
            LOGGER_DATASET.debug("Decreasing the size of dataset to fit the data exactly.")
            LOGGER_DATASET.info("new x_test shape: %s, original shape: %s",
                                (index_test, SIZE, SIZE, 3), x_test.shape)
            x_test.resize((index_test, SIZE, SIZE, 3))
            LOGGER_DATASET.info("new y_test shape: %s, original shape: %s",
                                (index_test, 1), y_test.shape)
            y_test.resize((index_test, 1))

        hf5_w["data/train"].attrs['num_of_images'] = index_train
        hf5_w["data/test"].attrs['num_of_images'] = index_test


def write_batch(x_data, y_data, x_batch, y_batch, index, data_type):
    """Function writes batch of data into hdf5 file. """
    if x_data.shape[0] <= index:
        LOGGER_DATASET.info("Increasing the size of %s datasets to fit the data.", data_type)
        LOGGER_DATASET.info("new x_%s shape: %s, original shape: %s",
                            data_type, (x_data.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3),
                            x_data.shape)
        x_data.resize((x_data.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3))
        LOGGER_DATASET.info("new y_%s shape: %s, original shape: %s",
                            data_type, (y_data.shape[0] + RESHAPE_STEP, 1), y_data.shape)
        y_data.resize((y_data.shape[0] + RESHAPE_STEP, 1))

    if len(x_batch) < BATCH_SIZE:
        x_data[index - len(x_batch):index] = np.stack(x_batch, axis=0)
        y_data[index - len(y_batch):index] = np.array(y_batch).reshape(len(y_batch), 1)
    else:
        x_data[index - BATCH_SIZE:index] = np.stack(x_batch, axis=0)
        y_data[index - BATCH_SIZE:index] = np.array(y_batch).reshape(BATCH_SIZE, 1)


def prepare_references(hf5_src, hf5_dst):
    """Returns list of references used during data spliting."""
    data_grp = hf5_src["data"]
    split_data_grp = hf5_dst.create_group("data")
    train = split_data_grp.create_group("train")
    test = split_data_grp.create_group("test")

    train.attrs['batch_size'] = TRAIN_BATCH_SIZE
    test.attrs['batch_size'] = TEST_BATCH_SIZE

    split_data_grp.attrs["nb_classes"] = data_grp.attrs["nb_classes"]
    split_data_grp.attrs['num_of_images'] = data_grp.attrs['num_of_images']

    train_est = int(data_grp.attrs['num_of_images'] * TRAIN_TEST_RATION)
    test_est = data_grp.attrs['num_of_images'] - train_est

    LOGGER_DATASET.info("creating datasets")
    x_train = train.create_dataset("x", (train_est, SIZE, SIZE, 3),
                                   maxshape=(None, SIZE, SIZE, 3))
    y_train = train.create_dataset("y", (train_est, 1), maxshape=(None, 1))
    x_test = test.create_dataset("x", (test_est, SIZE, SIZE, 3),
                                 maxshape=(None, SIZE, SIZE, 3))
    y_test = test.create_dataset("y", (test_est, 1), maxshape=(None, 1))

    indexes = []
    data = []

    for dataset in  data_grp.values():
        indexes.append([i for i in range(len(dataset))])
        data.append(dataset)

    return (x_train, y_train, x_test, y_test, data, indexes)


def get_image(file_path):
    """Function reads image from file path and if necessary applies several modification to
    match desired shape.
    """
    file_name = os.path.basename(file_path)
    try:
        raw_image = imread(file_path, mode="RGB")

        LOGGER_DATASET.debug("processing %s: %s", file_name, raw_image.shape)
        if raw_image.shape[0] == SIZE and raw_image.shape[1] == SIZE:
            pass
        elif raw_image.shape[0] > SIZE or raw_image.shape[1] > SIZE:
            LOGGER_DATASET.debug("resizing image %s: %s", file_name, raw_image.shape)
            resized_image = imresize(raw_image, get_percentage(raw_image))
            LOGGER_DATASET.debug("fixing image %s: %s", file_name, resized_image.shape)
            raw_image = fix_image(resized_image)
            LOGGER_DATASET.debug("resultin image %s: %s", file_name, raw_image.shape)
        else:
            raw_image = fix_image(raw_image)

        if raw_image.shape != (SIZE, SIZE, 3):
            LOGGER_DATASET.info("irregular, skipping %s: %s", file_path, raw_image.shape)
    except ValueError:
        LOGGER_DATASET.info("problem with %s: %s", file_path, raw_image.shape)
    except IndexError:
        LOGGER_DATASET.info("problem with fix_image, %s: %s", file_path, raw_image.shape)

    return raw_image


def get_percentage(image):
    """Returns percentage of original image that it needs to fit into
    (SIZE, SIZE, 3) size.
    """
    row = image.shape[0]
    col = image.shape[1]
    p_row = (100 * (SIZE - 1)) / row
    p_col = (100 * (SIZE - 1)) / col
    return p_row if p_row > p_col else p_col


def fix_image(img):
    """Normalize image"""
    new_img = []
    for i, channel in enumerate([img[:, :, ch] for ch in range(img.shape[2])]):
        new_img.append(fix_channel(channel, i))
    return np.dstack(new_img)


def fix_channel(channel, i):
    """Normalize channel"""
    LOGGER_DATASET.debug("Fixing channel %s", i)
    if channel.shape[0] < SIZE:
        channel = pad_x(channel)
    elif channel.shape[0] > SIZE:
        channel = crop_x(channel)

    if channel.shape[1] < SIZE:
        channel = pad_y(channel)
    elif channel.shape[1] > SIZE:
        channel = crop_y(channel)

    return channel


def pad_x(channel):
    """padd x direction of channel"""
    LOGGER_DATASET.debug("Padding X of channgel: %s", channel.shape)
    diff = int((SIZE - channel.shape[0]) / 2)
    odd = int((SIZE - channel.shape[0]) % 2)
    result = np.lib.pad(channel, ([diff, diff + odd], [0, 0]), 'constant', constant_values=(0))
    LOGGER_DATASET.debug("result of padding: %s", result.shape)
    return result


def pad_y(channel):
    """padd y direction of channel"""
    LOGGER_DATASET.debug("Padding Y of channgel: %s", channel.shape)
    diff = int((SIZE - channel.shape[1]) / 2)
    odd = int((SIZE - channel.shape[1]) % 2)
    result = np.lib.pad(channel, ([0, 0], [diff, diff + odd]), 'constant', constant_values=(0))
    LOGGER_DATASET.debug("result of padding: %s", result.shape)
    return result


def crop_x(channel):
    """crop x direction of channel"""
    LOGGER_DATASET.debug("Cropping X channgel: %s", channel.shape)
    diff = int((channel.shape[0] - SIZE) / 2)
    odd = int((channel.shape[0] - SIZE) % 2)
    result = channel[diff:channel.shape[0] - diff - odd, :]
    LOGGER_DATASET.debug("result of cropping: %s", result.shape)
    return result


def crop_y(channel):
    """crop y direction of channel"""
    LOGGER_DATASET.debug("Cropping Y channgel: %s", channel.shape)
    diff = int((channel.shape[1] - SIZE) / 2)
    odd = int((channel.shape[1] - SIZE) % 2)
    result = channel[:, diff:channel.shape[1] - diff - odd]
    LOGGER_DATASET.debug("result of cropping: %s", result.shape)
    return result


def generate_train_data(hf5_file_name, batch_size=None):
    """Generator that is providing infinite loop of testing dataset.
    Dataset is loaded from hdf5 file specified by file name. Size of
    each batch of data is either determined from parameter batch_size
    of from hdf5 file attribute.
    """
    with h5py.File(hf5_file_name, 'r') as hf5:
        grp = hf5["/data/train"]
        data_x = grp["x"]
        data_y = grp["y"]
        pos = 0
        size = data_x.shape[0]

        if batch_size:
            step = batch_size
        else:
            step = grp.attrs["batch_size"]

        while True:
            if pos + step <= size:
                batch_x = data_x[pos:pos + step, :, :, :]
                batch_y = data_y[pos:pos + step, :]
            else:
                temp = pos
                pos = (pos + step) - size
                batch_x = np.concatenate((data_x[0:pos, :, :, :], data_x[temp:size, :, :, :]))
                batch_y = np.concatenate((data_y[0:pos, :], data_y[temp:size, :]))

            undersized_batch_x = np.empty((step, TRAIN_SIZE, TRAIN_SIZE, 3))

            for index, image in enumerate(batch_x):
                undersized_batch_x[index, :, :, :] = generate_random_patch(image)

            yield (undersized_batch_x, batch_y)

            pos += step


def generate_test_data(hf5_file_name, batch_size=None):
    """Generator that is providing infinite loop of testing dataset.
    Dataset is loaded from hdf5 file specified by file name. Size of
    each batch of data is either determined from parameter batch_size
    of from hdf5 file attribute.
    """
    with h5py.File(hf5_file_name, 'r') as hf5:
        grp = hf5["/data/test"]
        data_x = grp["x"]
        data_y = grp["y"]
        pos = 0
        size = data_x.shape[0]
        if batch_size:
            step = batch_size
        else:
            step = grp.attrs["batch_size"]
        while True:
            if pos + step <= size:
                batch_x = data_x[pos:pos + step, :, :, :]
                batch_y = data_y[pos:pos + step, :]

            else:
                temp = pos
                pos = (pos + step) - size

                batch_x = np.concatenate((data_x[0:pos, :, :, :], data_x[temp:size, :, :, :]))
                batch_y = np.concatenate((data_y[0:pos, :], data_y[temp:size, :]))

            undersized_batch_x = np.empty((step, TRAIN_SIZE, TRAIN_SIZE, 3))

            for index, image in enumerate(batch_x):
                undersized_batch_x[index, :, :, :] = get_center_patch(image)

            yield (undersized_batch_x, batch_y)


            pos += step


def generate_random_indexes(indexes):
    """Generates random pair of indexes for all images """
    def get_most_frequent(indexes):
        """function for determining the index of most frequent category
        """
        longest = 0
        longest_index = -1
        for i, index in enumerate(indexes):
            if len(index) > longest:
                longest = len(index)
                longest_index = i
        return longest_index
    ind = [i for i in range(len(indexes))]
    count = 0
    while ind:
        if not count % 3:
            i = get_most_frequent(indexes)
            if indexes[i]:
                yield (i, indexes[i].pop(randint(0, len(indexes[i]) - 1)))
            else:
                ind.pop(i)
            count += 1
        else:
            i = randint(0, len(ind) - 1)
            if indexes[ind[i]]:
                yield (ind[i], indexes[ind[i]].pop(randint(0, len(indexes[ind[i]]) - 1)))
            else:
                ind.pop(i)
            count += 1


def generate_random_patch(image):
    """ Function returns random patch from original image. """
    x_rand = randint(0, DIFF)
    y_rand = randint(0, DIFF)
    patch = image[x_rand:TRAIN_SIZE+x_rand, y_rand:TRAIN_SIZE+y_rand, :]
    if randint(0, 1):
        patch = np.flip(patch, 1)
    return patch


def generate_patches(image, count):
    """ Function returns random list of patches from original image. """
    patches = []
    for _ in range(count):
        x_rand = randint(0, DIFF)
        y_rand = randint(0, DIFF)
        patch = image[x_rand:TRAIN_SIZE+x_rand, y_rand:TRAIN_SIZE+y_rand, :]
        if randint(0, 1):
            patch = np.flip(patch, 1)
        patches.append(patch)
    return patches


def get_center_patch(image):
    """Returns center patch from the image. """
    diff = int((SIZE-TRAIN_SIZE)/2)
    return image[diff:-diff, diff:-diff, :]


def main():
    """Main function"""
    load_dataset_from_images(SOURCE, HDF5_CAT)
    split_data(HDF5_CAT, HDF5_SPL)
    prepare_dataset_from_hf5(HDF5_SPL, HDF5_PRP)


if __name__ == "__main__":
    main()
