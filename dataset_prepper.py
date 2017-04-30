"This project will be used for preprocessing of ImageNet dataset for master thesis"
import glob
import os
import os.path

import h5py
import numpy as np
from keras.utils import np_utils
from scipy.misc import imresize, imsave
from scipy.ndimage import imread

from loggers import LOGGER_DATASET

SOURCE = os.path.expanduser("~/projects/ILSVRC/source_dataset/")
DESTINATION = os.path.expanduser("~/projects/ILSVRC/destination_dataset/")

HDF5_SRC = "../datasets/image_net_source.h5"
HDF5_DST = "../datasets/image_net.h5"

SIZE = 256
MAX_VAL = 255

def prepare_dataset_from_hf5(hf5_file_name):
    """TODO: Update
    Think how to make this as independet from the rest of the script as possible.
    Load data from CIFAR10 database."""

    with h5py.File(hf5_file_name, 'r') as hf5, h5py.File(HDF5_DST, 'w') as hf5_w:
        nb_classes = hf5["x_train"].attrs['nb_classes']
        size = hf5["x_train"].attrs['batch_sizes']
        test_batch_size = hf5["x_test"].attrs['batch_sizes']

        LOGGER_DATASET.info("Loading data from HDF5 file")
        x_train_dset = hf5_w.create_dataset("x_train", hf5["x_train"].shape, dtype=np.float32, chunks=True)
        y_train_dset = hf5_w.create_dataset("y_train", (hf5["y_train"].shape[0], nb_classes), chunks=True)
        x_test_dset = hf5_w.create_dataset("x_test", hf5["x_test"].shape, dtype=np.float32, chunks=True)
        y_test_dset = hf5_w.create_dataset("y_test", (hf5["y_test"].shape[0], nb_classes), chunks=True)

        x_train_dset.attrs["batch_size"] = size
        y_train_dset.attrs["nb_classes"] = nb_classes
        x_test_dset.attrs["batch_size"] = test_batch_size
        y_test_dset.attrs["nb_classes"] = nb_classes

        LOGGER_DATASET.info("Chunkification:")

        for index in range(0, hf5["x_train"].shape[0], size):
            print("processing chunk (%s:%s)" % (index, index+size))
            x_train_dset[index:index+size] = np.divide(hf5["x_train"][index:index+size], MAX_VAL)
            y_train_dset[index:index+size] = np_utils.to_categorical(hf5["y_train"][index:index+size], nb_classes)

        for index in range(0, hf5["x_test"].shape[0], size):
            print("processing chunk (%s:%s)" % (index, index+size))
            x_test_dset[index:index+size] = np.divide(hf5["x_test"][index:index+size], MAX_VAL)
            y_test_dset[index:index+size] = np_utils.to_categorical(hf5["y_test"][index:index+size], nb_classes)

        LOGGER_DATASET.info("finished")

        LOGGER_DATASET.info("dtype of dset:")
        print(x_train_dset.dtype)
        print(y_train_dset.dtype)
        print(x_test_dset.dtype)
        print(y_test_dset.dtype)

        LOGGER_DATASET.info("dtype of dset:")
        print(x_train_dset.shape)
        print(y_train_dset.shape)
        print(x_test_dset.shape)
        print(y_test_dset.shape)

        return  (x_train_dset[:], y_train_dset[:]), (x_test_dset[:], y_test_dset[:])


def load_dataset_from_hf5(hf5_file_name):
    """ TODO: update
    Load dataset from specified hdf5 file. """
    with h5py.File(hf5_file_name, 'r') as hf5:
        LOGGER_DATASET.info("Loading data from HDF5 file")
        return (hf5["x_train"][:], hf5["y_train"][:]), (hf5["x_test"][:], hf5["y_test"][:])


def load_dataset_from_images(source, hdf5_file_name):
    """Function loads images from source directory and loads them into
    numpy ndarrys.
    """
    with h5py.File(hdf5_file_name, 'w') as hf5:
        hf5.create_group("data")

        dataset_x = []
        dataset_y = []
        class_keys = {}

        # index = 0

        for dir_path in sorted(glob.glob(os.path.join(source, "*/"))):
            images = []
            classes = []

            dir_name = os.path.split(os.path.dirname(dir_path))[1]

            hf5.create_dataset("/data/%s" % dir_name, (1000, SIZE, SIZE, 3), maxshape=(None, SIZE, SIZE, 3))

            if not os.path.exists(DESTINATION + dir_name):
                os.mkdir(DESTINATION + dir_name)

            for index, file_path in enumerate(sorted(glob.glob(dir_path + "/*.JPEG"))):

                # if dir_name not in class_keys:
                #     class_keys[dir_name] = index
                #     index += 1

                try:
                    new_image = imread(file_path, mode="RGB")
                    if new_image.shape == (SIZE, SIZE, 3):
                        images.append(new_image)
                        classes.append(class_keys[dir_name])
                    else:
                        LOGGER_DATASET.info("%s: %s", file_path, new_image.shape)

                except ValueError:
                    LOGGER_DATASET.info("%s: %s", file_path, imread(file_path, mode="RGB").shape)

            dataset_x.append(images)
            dataset_y.append(classes)

        return split_data((dataset_x, dataset_y))


def convert_data_for_datset(source, destination):
    """Function resizes all images in source folder into (SIZE, SIZE, 3)
    format and saves it to destination folder.
    """
    for dir_path in sorted(glob.glob(os.path.join(source, "*/"))):

        dir_name = os.path.split(os.path.dirname(dir_path))[1]

        if not os.path.exists(DESTINATION + dir_name):
            os.mkdir(DESTINATION + dir_name)

        for file_path in sorted(glob.glob(dir_path + "/*.JPEG")):
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
                    LOGGER_DATASET.info("irregular, skipping %s: %s", file_name, raw_image.shape)
                    continue

            except ValueError:
                LOGGER_DATASET.info("problem with %s: %s", file_path, raw_image.shape)
            except IndexError:
                LOGGER_DATASET.info("Problem with fix_image, %s, %s", file_path, raw_image.shape)

            imsave(os.path.join(destination, dir_name, file_name), raw_image)
            LOGGER_DATASET.info("file  %s saved", file_name)


def get_percentage(image):
    """Returns percentage of original image that it needs to fit into
    (SIZE, SIZE, 3) size.
    """
    row = image.shape[0]
    col = image.shape[1]
    p_row = (100*(SIZE-1))/row
    p_col = (100*(SIZE-1))/col
    if p_row > p_col:
        return p_row
    else:
        return p_col


def split_data(data):
    """Splits data into train and test datesets."""
    (dataset_x, dataset_y) = data
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    i = 0
    for class_x, class_y in zip(dataset_x, dataset_y):
        for x_input, y_input in zip(class_x, class_y):
            if i < 10:
                x_train.append(x_input)
                y_train.append(y_input)
            else:
                x_test.append(x_input)
                y_test.append(y_input)
                i = 0
            i += 1
    x_train = np.stack(x_train, axis=0)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    x_test = np.stack(x_test, axis=0)
    y_test = np.array(y_test).reshape(len(y_test), 1)

    return (x_train, y_train), (x_test, y_test)

def save_data_to_hf5(data):
    """First attempt to save data to hdf5 data format."""

    (x_train, y_train), (x_test, y_test) = data

    with h5py.File(HDF5_SRC, 'w') as hf_w:
        hf_w.create_dataset("x_train", data=x_train)
        hf_w.create_dataset("y_train", data=y_train)
        hf_w.create_dataset("x_test", data=x_test)
        hf_w.create_dataset("y_test", data=y_test)


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
    result = np.lib.pad(channel, ([diff, diff+odd], [0, 0]), 'constant', constant_values=(0))
    LOGGER_DATASET.debug("result of padding: %s", result.shape)
    return result


def pad_y(channel):
    """padd y direction of channel"""
    LOGGER_DATASET.debug("Padding Y of channgel: %s", channel.shape)
    diff = int((SIZE - channel.shape[1]) / 2)
    odd = int((SIZE - channel.shape[1]) % 2)
    result = np.lib.pad(channel, ([0, 0], [diff, diff+odd]), 'constant', constant_values=(0))
    LOGGER_DATASET.debug("result of padding: %s", result.shape)
    return result


def crop_x(channel):
    """crop x direction of channel"""
    LOGGER_DATASET.debug("Cropping X channgel: %s", channel.shape)
    diff = int((channel.shape[0] - SIZE) / 2)
    odd = int((channel.shape[0] - SIZE) % 2)
    result = channel[diff:channel.shape[0]-diff-odd, :]
    LOGGER_DATASET.debug("result of cropping: %s", result.shape)
    return result


def crop_y(channel):
    """crop y direction of channel"""
    LOGGER_DATASET.debug("Cropping Y channgel: %s", channel.shape)
    diff = int((channel.shape[1] - SIZE) / 2)
    odd = int((channel.shape[1] - SIZE) % 2)
    result = channel[:, diff:channel.shape[1]-diff-odd]
    LOGGER_DATASET.debug("result of cropping: %s", result.shape)
    return result

def main():
    """Main function"""
    # convert_data_for_datset(SOURCE, DESTINATION)
    # save_data_to_hf5(HDF5_SRC)
    # (x_train, y_train), (x_test, y_test) = load_dataset_from_hf5(HDF5_DST)
    prepare_dataset_from_hf5(HDF5_SRC)


if __name__ == "__main__":
    main()
