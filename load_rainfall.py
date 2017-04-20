"""
Load rainfall data from the mongo DB
"""
from pathlib import Path
import pymongo
import numpy as np
from skimage.measure import block_reduce


db = pymongo.MongoClient().rainfall_sz  # mongodb database
NUM_TRAIN = 10000


def load_training_data_sklearn(t, height_span, image_size, downsample_size, limit=NUM_TRAIN):
    """
    load training data and make X flat for sklearn learners
    """
    training_X, training_Y = load_training_data(t=t, height_span=height_span, img_size=image_size, downsample_size=downsample_size, limit=limit)
    flatten_training_X = training_X.flatten().reshape(len(training_Y), -1)
    return flatten_training_X, training_Y


def load_training_data(t, height_span, img_size, downsample_size, limit=NUM_TRAIN, data_format="channels_last"):
    """
    load training data from the mongo db

    ### arguments:
        t: time slit no.
        height_span: a list of heghts needed
        limit: the limit of number of training samples (now always reading all the data for the first time and then store to disk)
        img_size: the side length of the image, centered at image centor point
        downsample_size: if > 1, is the downsample size for the image (func: np.max)
        data_format: channels_first or channels_last (always set to channels_last in CIKM competition)

    return X, y
    """
    print("loading training data... t: {}, h: {}, img_size: {}, downsample: {}, limit: {}".format(t, height_span, img_size, downsample_size, limit))
    if limit <= 0 or limit > NUM_TRAIN:
        limit = NUM_TRAIN
    if img_size <= 0 or img_size > 101:
        img_size = 101

    parameter_setting = '{}_{}_{}_{}'.format(t, height_span, img_size, downsample_size)
    cache_file_X = 'cached_training_data/X_{}.npy'.format(parameter_setting)
    cache_file_Y = 'cached_training_data/Y_{}.npy'.format(parameter_setting)

    if Path(cache_file_X).is_file() and Path(cache_file_Y).is_file():
        print("find cached raw data!")
        train_X = list(np.load(cache_file_X))
        train_Y = list(np.load(cache_file_Y))

    else:
        train_X = [0] * NUM_TRAIN
        train_Y = [0] * NUM_TRAIN

        for i in range(NUM_TRAIN):
            train_id = 'train_{}'.format(i + 1)
            if i % 200 == 0:
                print(i)
            X = []
            for h in height_span:
                record = db['train_t{}h{}'.format(t, h)].find_one({'train_id': train_id})
                rec_x = np.asarray(record['spatial_data']).reshape(101, 101).astype(int)
                if downsample_size > 1:
                    rec_x = block_reduce(image=rec_x, block_size=(downsample_size, downsample_size), func=np.max)
                idx_low = int(rec_x.shape[0] / 2) - int(img_size / 2)
                idx_high = int(rec_x.shape[0] / 2) + int((img_size + 1) / 2)
                train_Y[i] = record['rainfall']
                rec_x = rec_x[idx_low:idx_high, idx_low:idx_high]
                X.append(rec_x)
            train_X[i] = np.stack(X)

        train_X = np.asarray(train_X)
        train_Y = np.asarray(train_Y).reshape(-1, 1)  # reshape to column vector

        np.save(cache_file_X, train_X)
        np.save(cache_file_Y, train_Y)

    if data_format == 'channels_last':
        print("change channels to last")
        train_X = np.moveaxis(train_X, 1, -1)  # move height dimension to the last position

    return train_X[0:limit], train_Y[0:limit]


def load_training_data_sklearn_4_viewpoints(t, height_span, image_size, downsample_size, limit=NUM_TRAIN):
    """
    load training data and change to flatten X
    """
    training_Xs, training_Y = load_training_data_4_viewpoints(
        t=t, height_span=height_span, img_size=image_size, downsample_size=downsample_size, limit=limit)
    flatten_training_Xs = []
    for training_X in training_Xs:
        flatten_training_X = training_X.flatten().reshape(len(training_Y), -1)
        flatten_training_Xs.append(flatten_training_X)
    return flatten_training_Xs, training_Y


def load_training_data_4_viewpoints(t, height_span, img_size, downsample_size, limit=NUM_TRAIN):
    """
    load training data from the mongo db
    t: time slit no.
    height_span: a list of heghts needed
    limit: the limit of number of training samples
    img_size: the side length of the image, centered at image centor point
    downsample_size: if > 1, is the downsample size for the image (func: np.max)
    return X, y
    """
    print("loading training data with 4 viewpoints... t: {}, h: {}, img_size: {}, downsample: {}, limit: {}".format(
        t, height_span, img_size, downsample_size, limit))
    train_X, train_Y = load_training_data(t, height_span, img_size, downsample_size, limit)
    cut_point = int(img_size * 2.0 / 3.0)
    train_X1 = train_X[:, :, 0:cut_point, 0:cut_point]  # left up viewpoint
    train_X2 = train_X[:, :, (img_size - cut_point):, 0:cut_point]  # left down viewpoint
    train_X3 = train_X[:, :, 0:cut_point, (img_size - cut_point):]  # right up viewpoint
    train_X4 = train_X[:, :, (img_size - cut_point):, (img_size - cut_point):]  # right down viewpoint

    return [train_X1, train_X2, train_X3, train_X4], train_Y


def load_testA_data(t, height_span, img_size, downsample_size=1, limit=-1):
    """
    load training data from the mongo db
    t: time slit no.
    height_span: a list of heghts needed
    img_size: the side length of the image, centered at the point [50, 50]
    return X
    """
    if limit <= 0 or limit > 2000:
        limit = 2000
    if img_size <= 0 or img_size > 101:
        img_size = 101

    test_X = [0] * limit  # test data has 2000 records

    for i in range(1, limit + 1):
        test_id = "test_A{}".format(i)
        X = []
        for h in height_span:
            record = db['test_t{}h{}'.format(t, h)].find_one({'test_id': test_id})
            rec_x = np.asarray(record['spatial_data']).reshape(101, 101).astype(int)
            if downsample_size > 1:
                rec_x = block_reduce(image=rec_x, block_size=(downsample_size, downsample_size), func=np.max)
            idx_low = int(rec_x.shape[0] / 2) - int(img_size / 2)
            idx_high = int(rec_x.shape[0] / 2) + int(img_size / 2) + 1
            rec_x = rec_x[idx_low:idx_high, idx_low:idx_high]
            X.append(rec_x)
        test_X[i - 1] = np.stack(X)
    test_X = np.asarray(test_X)
    return test_X
