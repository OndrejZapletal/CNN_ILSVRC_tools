"This project will be used for preprocessing of ImageNet dataset for master thesis"
import glob
import os
import os.path

import h5py
import numpy as np
from keras.utils import np_utils
from scipy.misc import imresize
from scipy.ndimage import imread

from loggers import LOGGER_DATASET

SOURCE = os.path.expanduser("~/projects/ILSVRC/source_dataset/")
DESTINATION = os.path.expanduser("~/projects/ILSVRC/destination_dataset/")

HDF5_CAT = "../datasets/image_net_categorized.h5"
HDF5_SPL = "../datasets/image_net_split.h5"
HDF5_PRP = "../datasets/image_net_prepared.h5"

SIZE = 256
MAX_VAL = 255
RESHAPE_STEP = 300
AVR_IMAGES_PER_CLASS = 500
TRAIN_TEST_RATION = 0.9
TRAIN_BATCH_SIZE = 500
TEST_BATCH_SIZE = 500
BATCH_SIZE = 500
DELIMITER = "*"*30 + "\n"

def load_dataset_from_images(source, hdf5_file_name):
    """Function loads images from source directory and loads them into
    numpy ndarrys.
    """
    with h5py.File(hdf5_file_name, 'w') as hf5_w:
        print("%sLoading dataset from images\n%s" % (DELIMITER, DELIMITER))
        data_grp = hf5_w.create_group("data")
        data_grp.attrs["nb_classes"] = 0
        num_of_images = 0
        directories = sorted(glob.glob(os.path.join(source, "*/")))

        for index, dir_path in enumerate(directories, start=1):

            dir_name = os.path.split(os.path.dirname(dir_path))[1]

            LOGGER_DATASET.info("Processing directory %s (% of %s)", dir_name, index, len(directories))
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
                    dataset[index-BATCH_SIZE:index] = np.stack(image_batch, axis=0)
                    image_batch = []

                LOGGER_DATASET.debug("File %s saved into %s (%s of %s)",
                                     file_path, dataset.name, index_f, len(files))

            if len(image_batch) > 0:
                dataset[index-len(image_batch):index] = np.stack(image_batch, axis=0)

            # at the end of folder resize shape to exactly fit the number of images within dataset
            if dataset.shape[0] > index:
                dataset.resize((index, SIZE, SIZE, 3))

            # at the end of folder increase number of classes
            data_grp.attrs["nb_classes"] += 1
        data_grp.attrs['num_of_images'] = num_of_images


def split_data(hf5_file_name_src, hf5_file_name_dst):
    """Splits data into train and test datesets."""

    with h5py.File(hf5_file_name_src, 'r') as hf5_r, \
         h5py.File(hf5_file_name_dst, 'w') as hf5_w:

        print("%sSpliting date into train and test.\n%s" % (DELIMITER, DELIMITER))
        data_grp = hf5_r["data"]
        split_data_grp = hf5_w.create_group("split_data")
        train_grp = split_data_grp.create_group("train")
        test_grp = split_data_grp.create_group("test")

        train_grp.attrs['batch_size'] = TRAIN_BATCH_SIZE
        test_grp.attrs['batch_size'] = TEST_BATCH_SIZE
        split_data_grp.attrs["nb_classes"] = data_grp.attrs["nb_classes"]
        split_data_grp.attrs['num_of_images'] = data_grp.attrs['num_of_images']

        train_est = int(data_grp.attrs['num_of_images'] * TRAIN_TEST_RATION)
        test_est = data_grp.attrs['num_of_images'] - train_est

        LOGGER_DATASET.info("creating datasets")
        x_train = train_grp.create_dataset("x_train", (train_est, SIZE, SIZE, 3), maxshape=(None, SIZE, SIZE, 3))
        y_train = train_grp.create_dataset("y_train", (train_est, 1), maxshape=(None, 1))
        x_test = test_grp.create_dataset("x_test", (test_est, SIZE, SIZE, 3), maxshape=(None, SIZE, SIZE, 3))
        y_test = test_grp.create_dataset("y_test", (test_est, 1), maxshape=(None, 1))

        index_train = 0
        index_test = 0

        LOGGER_DATASET.info("staring splitting")
        datasets = data_grp.values()
        for category, dataset in enumerate(datasets):
            image_batch_train_x = []
            image_batch_train_y = []
            image_batch_test_x = []
            image_batch_test_y = []
            LOGGER_DATASET.info("splitting dataset %s (%s of %s)", dataset.name, category+1, len(datasets))
            for index, image in enumerate(dataset):
                if not index % 10:
                    check_train_size(x_train, y_train, index_train)
                    image_batch_train_x.append(image)
                    image_batch_train_y.append(category)
                    index_train += 1

                    if len(image_batch_train_x) == BATCH_SIZE:
                        x_train[index_train-BATCH_SIZE:index_train] = np.stack(image_batch_train_x, axis=0)
                        y_train[index_train-BATCH_SIZE:index_train] = np.array(image_batch_train_y).reshape(BATCH_SIZE, 1)
                        image_batch_train_x = []
                        image_batch_train_y = []

                else:
                    check_test_size(x_test, y_test, index_test)
                    image_batch_test_x.append(image)
                    image_batch_test_y.append(category)
                    index_test += 1

                    if len(image_batch_test_x) == BATCH_SIZE:
                        x_test[index_test-BATCH_SIZE:index_test] = np.stack(image_batch_test_x, axis=0)
                        y_test[index_test-BATCH_SIZE:index_test] = np.array(image_batch_test_y).reshape(BATCH_SIZE, 1)
                        image_batch_test_x = []
                        image_batch_test_y = []

            if len(image_batch_train_x) > 0:
                x_train[index_train-len(image_batch_train_x):index_train] = np.stack(image_batch_train_x, axis=0)
                y_train[index_train-len(image_batch_train_y):index_train] = np.array(image_batch_train_y).reshape(len(image_batch_train_y), 1)

            if len(image_batch_test_x) > 0:
                x_test[index_test-len(image_batch_test_x):index_test] = np.stack(image_batch_test_x, axis=0)
                y_test[index_test-len(image_batch_test_y):index_test] = np.array(image_batch_test_y).reshape(len(image_batch_test_y), 1)

        LOGGER_DATASET.info("Decreasing the size of datasets to fit the data exactly.")
        if x_train.shape[0] > index_train:
            LOGGER_DATASET.info("new x_train shape: %s, original shape: %s", (index_train, SIZE, SIZE, 3), x_train.shape)
            x_train.resize((index_train, SIZE, SIZE, 3))
            LOGGER_DATASET.info("new y_train shape: %s, original shape: %s", (index_train, 1), y_train.shape)
            y_train.resize((index_train, 1))

        if x_test.shape[0] > index_test:
            LOGGER_DATASET.info("new x_test shape: %s, original shape: %s", (index_test, SIZE, SIZE, 3), x_test.shape)
            x_test.resize((index_test, SIZE, SIZE, 3))
            LOGGER_DATASET.info("new y_test shape: %s, original shape: %s", (index_test, 1), y_test.shape)
            y_test.resize((index_test, 1))


def check_train_size(x_train, y_train, index_train):
    if x_train.shape[0] <= index_train:

        LOGGER_DATASET.info("Increasing the size of train datasets to fit the data.")
        LOGGER_DATASET.info("new x_train shape: %s, original shape: %s",
                            (x_train.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3) , x_train.shape)
        x_train.resize((x_train.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3))
        LOGGER_DATASET.info("new y_train shape: %s, original shape: %s",
                            (y_train.shape[0] + RESHAPE_STEP, 1),
                            y_train.shape)
        y_train.resize((y_train.shape[0] + RESHAPE_STEP, 1))


def check_test_size(x_test, y_test, index_test):
    if x_test.shape[0] <= index_test:
        LOGGER_DATASET.info("Increasing the size of test datasets to fit the data.")
        LOGGER_DATASET.info("new x_test shape: %s, original shape: %s",
                            (x_test.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3) , x_test.shape)
        x_test.resize((x_test.shape[0] + RESHAPE_STEP, SIZE, SIZE, 3))
        LOGGER_DATASET.info("new y_test shape: %s, original shape: %s",
                            (y_test.shape[0] + RESHAPE_STEP, 1),
                            y_test.shape)
        y_test.resize((y_test.shape[0] + RESHAPE_STEP, 1))


def prepare_dataset_from_hf5(hf5_file_source, hf5_file_destination):
    """TODO: Update
    Think how to make this as independet from the rest of the script as possible.
    Load data from CIFAR10 database."""

    with h5py.File(hf5_file_source, 'r') as hf5_r, h5py.File(hf5_file_destination, 'w') as hf5_w:
        print("%sPreparing dataset\n%s" % (DELIMITER, DELIMITER))
        data_grp_dst = hf5_w.create_group("prepared_data")
        test_grp_dst = hf5_w.create_group("/prepared_data/test")

        nb_classes = hf5_r["split_data"].attrs['nb_classes']
        size = hf5_r["/split_data/train"].attrs['batch_size']
        # test_batch_size = hf5_r["/split_data/test"].attrs['batch_size']
        train_grp_dst.attrs['batch_size'] = hf5_r["/split_data/train"].attrs['batch_size']
        test_grp_dst.attrs['batch_size'] = hf5_r["/split_data/test"].attrs['batch_size']

        x_train_src = hf5_r["/split_data/train/x_train"]
        y_train_src = hf5_r["/split_data/train/y_train"]
        x_test_src = hf5_r["/split_data/test/x_test"]
        y_test_src = hf5_r["/split_data/test/y_test"]

        LOGGER_DATASET.info("Loading data from HDF5 file")
        x_train_dst = train_grp_dst.create_dataset("x_train", x_train_src.shape, dtype=np.float32, chunks=True)
        y_train_dst = train_grp_dst.create_dataset("y_train", (y_train_src.shape[0], nb_classes), chunks=True)
        x_test_dst = test_grp_dst.create_dataset("x_test", x_test_src.shape, dtype=np.float32, chunks=True)
        y_test_dst = test_grp_dst.create_dataset("y_test", (y_test_src.shape[0], nb_classes), chunks=True)

        print(x_train_src.shape)
        print(y_train_src.shape)
        print(x_test_src.shape)
        print(y_test_src.shape)
        print(x_train_dst.shape)
        print(y_train_dst.shape)
        print(x_test_dst.shape)
        print(y_test_dst.shape)

        LOGGER_DATASET.info("Chunkification:")

        for index in range(0, x_train_src.shape[0], size):
            print("processing chunk [%s:%s]" % (index, index+size))
            x_train_dst[index:index+size] = np.divide(x_train_src[index:index+size], MAX_VAL)
            y_train_dst[index:index+size] = np_utils.to_categorical(y_train_src[index:index+size], nb_classes)

        for index in range(0, x_test_src.shape[0], size):
            print("processing chunk [%s:%s]" % (index, index+size))
            x_test_dst[index:index+size] = np.divide(x_test_src[index:index+size], MAX_VAL)
            y_test_dst[index:index+size] = np_utils.to_categorical(y_test_src[index:index+size], nb_classes)

        LOGGER_DATASET.info("finished")

        LOGGER_DATASET.info("dtype of dst:")
        print(x_train_dst.dtype)
        print(y_train_dst.dtype)
        print(x_test_dst.dtype)
        print(y_test_dst.dtype)

        LOGGER_DATASET.info("dtype of dset:")
        print(x_train_dst.shape)
        print(y_train_dst.shape)
        print(x_test_dst.shape)
        print(y_test_dst.shape)

        return  (x_train_dst[:], y_train_dst[:]), (x_test_dst[:], y_test_dst[:])


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
    p_row = (100*(SIZE-1))/row
    p_col = (100*(SIZE-1))/col
    if p_row > p_col:
        return p_row
    else:
        return p_col



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


def generate_train_data(hf5_file_name, batch_size=None):
    """Generator that is providing infinite loop of training dataset.
    Dataset is loaded from hdf5 file specified by file name. Size of
    each batch of data is either determined from parameter batch_size
    of from hdf5 file attribute.
    """
    with h5py.File(hf5_file_name, 'r') as hf5:
        train_grp = hf5["/prepared_data/train"]
        data_x = train_grp["x_train"]
        data_y = train_grp["y_train"]
        pos = 0
        size = data_x.shape[0]
        if batch_size:
            step = batch_size
        else:
            step = train_grp.attrs["batch_size"]
        while True:
            if pos + step <= size:
                yield (data_x[pos:pos+step],
                       data_y[pos:pos+step])
            else:
                temp = pos
                pos = (pos + step) - size
                yield (np.concatenate((data_x[0:pos], data_x[temp:size])),
                       np.concatenate((data_y[0:pos], data_y[temp:size])))
            pos += step


def generate_test_data(hf5_file_name, batch_size=None):
    """Generator that is providing infinite loop of testing dataset.
    Dataset is loaded from hdf5 file specified by file name. Size of
    each batch of data is either determined from parameter batch_size
    of from hdf5 file attribute.
    """
    with h5py.File(hf5_file_name, 'r') as hf5:
        test_grp = hf5["/prepared_data/test"]
        data_x = test_grp["x_test"]
        data_y = test_grp["y_test"]
        pos = 0
        size = data_x.shape[0]
        if batch_size:
            step = batch_size
        else:
            step = test_grp.attrs["batch_size"]
        while True:
            if pos + step <= size:
                yield (data_x[pos:pos+step],
                       data_y[pos:pos+step])
            else:
                temp = pos
                pos = (pos + step) - size
                yield (np.concatenate((data_x[0:pos], data_x[temp:size])),
                       np.concatenate((data_y[0:pos], data_y[temp:size])))
            pos += step

def main():
    """Main function"""
    load_dataset_from_images(SOURCE, HDF5_CAT)
    split_data(HDF5_CAT, HDF5_SPL)
    prepare_dataset_from_hf5(HDF5_SPL, HDF5_PRP)


if __name__ == "__main__":
    main()
