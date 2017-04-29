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

SOURCE = "/home/ondrej-zapletal/projects/ILSVRC/source_dataset/"
DESTINATION = "/home/ondrej-zapletal/projects/ILSVRC/destination_dataset/"

HDF5_SRC = "../configuration/image_net_source.h5"
HDF5_DST = "../configuration/image_net.h5"

SIZE = 256


def load_dataset_from_h5f(h5f_name):
    """TODO:
    Think how to make this as independet from the rest of the script as possible.
    Load data from CIFAR10 database."""

    with h5py.File(h5f_name, 'r') as h5f:
        nb_classes = 20
        max_val = 255

        h5f_w = h5py.File(HDF5_DST, 'w')

        LOGGER_DATASET.info("Loading data from HDF5 file")
        x_train = h5f["x_train"][:]
        LOGGER_DATASET.info("Loaded x_train")
        y_train = h5f["y_train"][:]
        LOGGER_DATASET.info("Loaded y_train")
        x_test = h5f["x_test"][:]
        LOGGER_DATASET.info("Loaded x_test")
        y_test = h5f["y_test"][:]
        LOGGER_DATASET.info("Loaded y_test")

        LOGGER_DATASET.info("Cloning datasets")
        x_train_dset = h5f_w.create_dataset("x_train", data=x_train, chunks=True)
        y_train_dset = h5f_w.create_dataset("y_train", data=y_train, chunks=True)
        x_test_dset = h5f_w.create_dataset("x_test", data=x_test, chunks=True)
        y_test_dset = h5f_w.create_dataset("y_test", data=y_test, chunks=True)
        LOGGER_DATASET.info("finished")

        print(x_train_dset.name)
        print(x_train_dset.shape)

        x_train_dset = x_train_dset.astype('float32')
        x_test_dset = x_test_dset.astype('float32')

        x_train_dset = x_train_dset[:] / max_val
        x_test_dset = x_test_dset[:] / max_val

        y_train_dset = np_utils.to_categorical(y_train_dset[:], nb_classes)
        y_test_dset = np_utils.to_categorical(y_test_dset[:], nb_classes)

        LOGGER_DATASET.debug("Training data X shape %s", x_train_dset.shape)
        LOGGER_DATASET.debug("Testing data X shape %s", x_test_dset.shape)
        LOGGER_DATASET.debug("Training data Y shape %s", y_train_dset.shape)
        LOGGER_DATASET.debug("Testing data Y shape %s", y_test_dset.shape)

        return  (x_train_dset[:], y_train_dset[:]), (x_test_dset[:], y_test_dset[:])


def load_dataset_from_disk():
    """Function loads images from source directory and loads them into
    numpy ndarrys.
    """
    dataset_x = []
    dataset_y = []
    class_keys = {}

    index = 0
    for dir_path in sorted(glob.glob(os.path.join(DESTINATION, "*/"))):
        images = []
        classes = []

        dir_name = os.path.split(os.path.dirname(dir_path))[1]

        if not os.path.exists(DESTINATION + dir_name):
            os.mkdir(DESTINATION + dir_name)

        for file_path in sorted(glob.glob(dir_path + "/*.JPEG")):

            if dir_name not in class_keys:
                class_keys[dir_name] = index
                index += 1

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


def fix_data():
    """Function resizes all images in source folder into (SIZE, SIZE, 3)
    format and saves it to destination folder.
    """
    for dir_path in sorted(glob.glob(os.path.join(SOURCE, "*/"))):

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

            imsave(os.path.join(DESTINATION, dir_name, file_name), raw_image)
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

def save_data(data):
    """First attempt to save data to hdf5 data format."""

    (x_train, y_train), (x_test, y_test) = data

    with h5py.File(HDF5_DST, 'w') as hf_w:
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
    # fix_data()
    # save_data()
    (x_train, y_train), (x_test, y_test) = load_dataset_from_h5f(HDF5_SRC)



if __name__ == "__main__":
    main()
