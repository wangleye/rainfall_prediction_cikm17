"""
Predict rainfall for the CIKM 2017 Competition
"""
import numpy as np
from keras.models import Sequential, load_model
from keras import backend, regularizers
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import keras.layers.recurrent as recurrent_unit
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn import metrics, ensemble
from sklearn.externals import joblib
import xgboost as xgb
# import pandas as pd
from load_rainfall import load_training_data, load_testA_data
from load_rainfall import load_training_data_sklearn, load_training_data_sklearn_4_viewpoints
from residual_blocks2 import building_residual_block
# from matplotlib import pyplot
# import plotly.plotly as py
# import plotly.graph_objs as go


NORM_X = 215.0
NORM_Y = 70.0


def rmse(Y, Y_pred):
    return np.sqrt(metrics.mean_squared_error(Y, Y_pred))


def CnnModel(input_shape, rweight=0.1):
    """
    simple CNN model structure with reference to LeNET
    """
    our_model = Sequential()
    print(input_shape)
    our_model.add(Conv2D(256, (5, 5), padding='same', activation='relu', input_shape=input_shape))
    our_model.add(Conv2D(256, (5, 5), padding='same', activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
    our_model.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
    our_model.add(MaxPooling2D(pool_size=(2, 2)))
    our_model.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
    our_model.add(Conv2D(512, (5, 5), padding='same', activation='relu'))
    our_model.add(Flatten())
    our_model.add(Dense(256, kernel_regularizer=regularizers.l2(rweight), activation='relu'))
    our_model.add(Dense(256, kernel_regularizer=regularizers.l2(rweight), activation='relu'))
    # our_model.add(Dropout(0.3))
    our_model.add(Dense(256, kernel_regularizer=regularizers.l2(rweight), activation='relu'))    
    our_model.add(Dropout(0.4))
    our_model.add(Dense(1, activation='sigmoid'))
    our_model.compile(loss='mean_squared_error',
                      optimizer='adadelta')
    return our_model


def ResModel(input_shape, n_channels=128):
    """
    residual model structure:
    input +
    [conv 3*3*n_channels + conv 3*3*n_channels] * num_res_blocks +
    flatten +
    sigmoid
    """
    our_model = Sequential()
    num_res_clocks = 6
    kernel_size = (3, 3)
    # build first layer
    our_model.add(Conv2D(n_channels, (3, 3), activation='relu', input_shape=input_shape))
    for _ in range(num_res_clocks):
        our_model.add(building_residual_block((input_shape[0], input_shape[1], n_channels),
                                              n_channels, kernel_size))
    our_model.add(Conv2D(2, (3, 3), activation='relu', input_shape=input_shape))
    our_model.add(Flatten())
    our_model.add(Dropout(0.4))
    our_model.add(Dense(1))
    our_model.compile(loss='mean_squared_error',
                      optimizer='adam')
    return our_model


def RnnModel(input_shape, n_channels=1024, rnn_unit=recurrent_unit.GRU, deep_level=7):
    our_model = Sequential()
    our_model.add(rnn_unit(units=n_channels, input_shape=input_shape, activation='relu', return_sequences=True))
    our_model.add(Dense(n_channels, activation='relu'))
    for _ in range(deep_level - 2):
        our_model.add(rnn_unit(units=n_channels, activation='relu', return_sequences=True))
        our_model.add(Dense(n_channels, activation='relu'))
    our_model.add(rnn_unit(units=n_channels, activation='relu', return_sequences=True))
    our_model.add(Flatten())
    our_model.add(Dense(n_channels, activation='relu'))
    our_model.add(Dropout(0.5))       
    our_model.add(Dense(1, activation='sigmoid'))
    our_model.compile(loss='mean_squared_error', optimizer='adadelta')

    return our_model


def ConvLstmModel(input_shape, deep_level=5, filters=64, rweight=0.001):
    """
    convolutional LSTM model
    deep_level: number of levels of conv lstm
    """
    our_model = Sequential()
    our_model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
                             input_shape=input_shape, padding='same',
                             return_sequences=True))
    for _ in range(deep_level - 2):
        our_model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
                                 padding='same',
                                 return_sequences=True))
    our_model.add(ConvLSTM2D(filters=filters, kernel_size=(3, 3),
                             padding='same',
                             return_sequences=False))
#    our_model.add(Conv3D(filters=input_shape[-1], kernel_size=(3, 3, 3),
#                         activation='relu', padding='same'))
    our_model.add(Flatten())
    our_model.add(Dense(filters*2, activation='relu', kernel_regularizer=regularizers.l2(rweight)))
    our_model.add(Dense(filters*2, activation='relu', kernel_regularizer=regularizers.l2(rweight)))
    our_model.add(Dropout(0.3))
    our_model.add(Dense(1, activation='relu'))
    our_model.compile(loss='mean_squared_error', optimizer='adadelta')

    return our_model


def StackingConvRNN(input_shape, rnn_level=3, filters=64, rweight=0.1):
    """
    Using Conv network over RNN unit layer
    input_shape: [n_timestep, img_size, img_size, n_channels]
    """
    our_model = Sequential()
    our_model.add(TimeDistributed(Conv2D(filters, (3, 3), padding='same', activation='relu'), input_shape=input_shape))
    our_model.add(TimeDistributed(BatchNormalization()))
    our_model.add(TimeDistributed(MaxPooling2D((2, 2))))
    our_model.add(TimeDistributed(Conv2D(filters, (3, 3), padding='same', activation='relu')))
    our_model.add(TimeDistributed(BatchNormalization()))
    our_model.add(TimeDistributed(Conv2D(filters, (3, 3), padding='same', activation='relu')))
    our_model.add(TimeDistributed(BatchNormalization()))
    our_model.add(TimeDistributed(MaxPooling2D((2, 2))))
    our_model.add(TimeDistributed(Flatten()))
    # our_model.add(TimeDistributed(Dropout(0.4)))
    for _ in range(rnn_level):
        our_model.add(recurrent_unit.GRU(filters, return_sequences=True))
        our_model.add(Dense(filters))
    our_model.add(Flatten())
    our_model.add(Dense(filters, kernel_regularizer=regularizers.l2(rweight)))
    our_model.add(Dropout(0.4))
    our_model.add(Dense(1, activation='relu'))
    our_model.compile(loss='mean_squared_error', optimizer='adadelta')

    return our_model


def augment_training_data(X, Y, image_size, mode):
    """
    for each matrix, random do flip to generate a new data
    mode: 'flattern' or 'image'
    """
    print('data augmentation...')
    augment_X = np.zeros(X.shape)
    augment_Y = np.zeros(Y.shape)

    for i, _ in enumerate(Y):
        ori_feature = X[i].reshape(image_size, image_size, -1)
        new_feature = np.zeros(ori_feature.shape)

        # do matrix flip to generate new training data
        for j in range(ori_feature.shape[0]):
            new_feature[j] = np.flipud(ori_feature[j])
            new_feature[j] = np.fliplr(ori_feature[j])

        if mode == 'flatten':
            augment_X[i] = new_feature.reshape(1, -1)
        elif mode == 'image':
            augment_X[i] = new_feature
        augment_Y[i] = Y[i]

        if i % 200 == 0:
            print("augmenting data {}".format(i), end='\r')
    print("\naugmenting done!")

    return np.concatenate((X, augment_X)), np.concatenate((Y, augment_Y))


def cross_validataion_avg_aggregate(t_span, height_span, image_size, learner, augment, downsample_size, test_ratio=0.2, limit=10000):
    K = int(1 / test_ratio)
    Y_pred_collection = None
    for t in t_span:
        X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        Y_1D = Y.reshape(-1)

        k_fold = KFold(K)
        Y_pred = np.zeros((len(Y), 1))
        for k, (train, test) in enumerate(k_fold.split(X, Y_1D)):
            if not augment:
                learner.fit(X[train], Y_1D[train])
            else:
                augment_X, augment_Y = augment_training_data(X[train], Y_1D[train], image_size, mode='flatten')
                learner.fit(augment_X, augment_Y)
            Y_pred[test] = learner.predict(X[test]).reshape(-1, 1)
            print("cv {} rmse: {}".format(k, rmse(Y_pred[test], Y[test])))

        if Y_pred_collection is None:
            Y_pred_collection = Y_pred
        else:
            Y_pred_collection = np.concatenate((Y_pred_collection, Y_pred), axis=1)

        avg_Y_pred = np.mean(Y_pred_collection, axis=1)
        print("t:{} h:{} rmse:{}".format(t, height_span, rmse(Y, Y_pred)))
        print("avg rmse:{}".format(rmse(Y, avg_Y_pred)))

    return Y_pred_collection, Y


def residual_error_regression(Y_pred_collection, Y, learner):
    """
    regression on the residual error to adjust the prediction on final result
    """
    avg_Y_pred = np.mean(Y_pred_collection, axis=1, keepdims=True)
    residual_Y = Y - avg_Y_pred
    residual_Y_pred = Y_pred_collection - avg_Y_pred
    learner.fit(residual_Y_pred, residual_Y)
    return learner


def cross_validataion_cnn(t_span, height_span, image_size, downsample_size, cnn_model, initial_weights, augment, test_ratio=0.2, limit=10000):
    """
    cross validation for cnn-like models
    """
    K = int(1 / test_ratio)
    Y_pred_collection = None
    for t in t_span:
        X, Y = load_training_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        X = X/NORM_X
        Y = Y/NORM_Y
        # X, Y = preprocessing_data(X, Y)

        k_fold = KFold(K)
        Y_pred = np.zeros((len(Y), 1))
        for k, (train, test) in enumerate(k_fold.split(X, Y)):
            reset_weights(cnn_model, initial_weights)
            train_X, train_Y = preprocessing_data(X[train], Y[train])
            test_X, test_Y = X[test], Y[test]
            if augment:
                train_X, train_Y = augment_training_data(train_X, train_Y, image_size, mode='image')
            early_stop = EarlyStopping(monitor='loss', patience=0)
            cnn_model.fit(train_X, train_Y, batch_size=32, epochs=200, verbose=1, validation_data=(test_X, test_Y), callbacks=[early_stop, ])
            Y_pred[test] = cnn_model.predict(test_X).reshape(-1, 1)
            print("cv {} rmse: {}".format(k, rmse(Y_pred[test] * NORM_Y, Y[test] * NORM_Y)))

        if Y_pred_collection is None:
            Y_pred_collection = Y_pred
        else:
            Y_pred_collection = np.concatenate((Y_pred_collection, Y_pred), axis=1)

        avg_Y_pred = np.mean(Y_pred_collection, axis=1)
        print("t:{} h:{} rmse:{}".format(t, height_span, rmse(Y, Y_pred)))
        print("avg rmse:{}".format(rmse(Y, avg_Y_pred)))

    return avg_Y_pred, Y


def cross_validataion_convlstm(t_span, height_span, image_size, downsample_size, convlstm_model, test_ratio=0.2, limit=10000):
    """
    cross validation on rnn+cnn models
    """
    K = int(1 / test_ratio)
    k_fold = KFold(K)
    time_length = len(t_span)
    channels = len(height_span)
    Xs = np.zeros((limit, time_length, image_size, image_size, channels))
    t_span.sort()
    for idx, t in enumerate(t_span):
        X, Y = load_training_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        X = X.reshape((limit, 1, image_size, image_size, channels))
        for i in range(limit):
            Xs[i, idx] = X[i]

    Xs = Xs / NORM_X
    Y = Y / NORM_Y

    Y_pred = np.zeros((limit, 1))
    for k, (train, test) in enumerate(k_fold.split(Xs, Y)):
        # reset_weights(convlstm_model, initial_weights)
        convlstm_model.fit(Xs[train], Y[train], batch_size=64, epochs=200, verbose=1, validation_data=(Xs[test], Y[test]))
        Y_pred[test] = convlstm_model.predict(Xs[test]).reshape(-1, 1)
        print("cv {} rmse: {}".format(k, NORM_Y * rmse(Y_pred[test], Y[test])))

    print("overall rmse: {}".format(NORM_Y * rmse(Y, Y_pred)))

    return Y_pred, Y


def preprocessing_data(X, Y):

    # remove the data with missing X or too large Y (abnormal values)
    neg_idxs = np.unique(np.argwhere(X < 0)[:, 0])
    large_idxs = np.unique(np.argwhere(Y > 1)[:, 0])
    merge_idx = np.unique(np.concatenate((neg_idxs, large_idxs)))
    mask = np.ones(len(Y), dtype=bool)
    mask[merge_idx] = False

    return X[mask], Y[mask]


def cross_validataion_rnn(t_span, height_span, image_size, downsample_size, rnn_model, initial_weights, test_ratio=0.2, limit=10000):
    """
    cross_validation on rnn models
    """
    K = int(1 / test_ratio)
    time_length = len(t_span)
    channels = len(height_span)
    Xs = np.zeros((limit, time_length, channels * image_size * image_size))
    t_span.sort()
    for idx, t in enumerate(t_span):
        X, Y = load_training_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        X = X.reshape((limit, 1, -1))
        for i in range(limit):
            Xs[i, idx] = X[i]

    Xs, Y = preprocessing_data(X, Y)

    k_fold = KFold(K)
    Y_pred = np.zeros((limit, 1))
    for k, (train, test) in enumerate(k_fold.split(Xs, Y)):
        reset_weights(rnn_model, initial_weights)
        rnn_model.fit(Xs[train], Y[train], batch_size=32, epochs=50, verbose=1, validation_data=(Xs[test], Y[test]))
        Y_pred[test] = rnn_model.predict(Xs[test]).reshape(-1, 1)
        print("cv {} rmse: {}".format(k, rmse(Y_pred[test], Y[test])))

    print("overall rmse: {}".format(rmse(Y, Y_pred)))

    return Y_pred, Y


def time_sensitive_validataion_avg_aggregate(
        t_span, height_span, image_size, learner, augment, downsample_size, residual_adjust=None, test_ratio=0.2, limit=10000):
    """
    use the first part of training data for training and the last part of the data for validation
    holdout is the percentage of the validation part
    """
    Y_pred_collection = None
    for t in t_span:
        X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        Y_1D = Y.reshape(-1)
        num_of_recs = len(Y_1D)
        train = range(int(num_of_recs * (1 - test_ratio)))
        test = range(int(num_of_recs * (1 - test_ratio)), num_of_recs)
        if not augment:
            X_train, Y_train = X[train], Y_1D[train]
        else:
            X_train, Y_train = augment_training_data(X[train], Y_1D[train], image_size, mode='flatten')
        learner.fit(X_train, Y_train)
        Y_pred = learner.predict(X[test]).reshape(-1, 1)

        if Y_pred_collection is None:
            Y_pred_collection = Y_pred
        else:
            Y_pred_collection = np.concatenate((Y_pred_collection, Y_pred), axis=1)

        avg_Y_pred = np.mean(Y_pred_collection, axis=1).reshape(-1, 1)
        print("t:{} h:{} rmse:{}".format(t, height_span, rmse(Y[test], Y_pred)))
        print("avg rmse:{}".format(rmse(Y[test], avg_Y_pred)))

    if residual_adjust is not None:
        residual_Y_pred = Y_pred_collection - avg_Y_pred
        residual_adjust_amount = residual_adjust.predict(residual_Y_pred).reshape(-1, 1)
        print(residual_adjust_amount.shape)
        adjust_avg_Y_pred = avg_Y_pred + residual_adjust_amount
        print("adjusted avg rmse:{}".format(rmse(Y[test], adjust_avg_Y_pred)))
        print(avg_Y_pred)
        print(residual_adjust_amount)

    np.savetxt("result_cache.txt", np.concatenate((Y[test], avg_Y_pred, Y_pred_collection), axis=1), fmt="%.5f")

    return avg_Y_pred, Y[test]


def time_sensitive_validataion_avg_aggregate_4_viewpoints(t_span, height_span, image_size, learner, augment, downsample_size, holdout=0.1):
    """
    use the first part of training data for training and the last part of the data for validation
    holdout is the percentage of the validation part
    """
    Y_pred_collection = None
    for t in t_span:
        Xs, Y = load_training_data_sklearn_4_viewpoints(
            t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=10000)
        Y_1D = Y.reshape(-1)
        num_of_recs = len(Y_1D)
        train = range(int(num_of_recs * (1 - holdout)))
        test = range(int(num_of_recs * (1 - holdout)), num_of_recs)
        Y_pred_4_viewpoints = np.zeros((len(test), 4))
        for idx, X in enumerate(Xs):
            if not augment:
                learner.fit(X[train], Y_1D[train])
            else:
                augment_X, augment_Y = augment_training_data(X[train], Y_1D[train], image_size, mode='flatten')
                learner.fit(augment_X, augment_Y)
            Y_pred_4_viewpoints[:, idx] = learner.predict(X[test]).reshape(-1)
            print("view point {} rmse: {}".format(idx + 1, rmse(Y_pred_4_viewpoints[:, idx], Y[test])))

        Y_pred = np.mean(Y_pred_4_viewpoints, axis=1).reshape(-1, 1)
        print("t:{} h:{} rmse:{}".format(t, height_span, rmse(Y[test], Y_pred)))
        if Y_pred_collection is None:
            Y_pred_collection = Y_pred
        else:
            Y_pred_collection = np.concatenate((Y_pred_collection, Y_pred), axis=1)
        avg_Y_pred = np.mean(Y_pred_collection, axis=1)
        print("avg rmse:{}".format(rmse(Y[test], avg_Y_pred)))

    return avg_Y_pred, Y[test]


def reset_weights(cnn_model, weights=None):
    if weights is None:
        weights = cnn_model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    cnn_model.set_weights(weights)


def time_sensitive_validataion_cnn(t_span, height_span, image_size, downsample_size, cnn_models, initial_weights, augment, holdout=0.1):
    """
    use the first part of training data for training and the last part of the data for validation
    holdout is the percentage of the validation part
    """
    Y_pred_collection = None
    for t in t_span:
        X, Y = load_training_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=10000)
        num_of_recs = len(Y)
        train = range(int(num_of_recs * (1 - holdout)))
        test = range(int(num_of_recs * (1 - holdout)), num_of_recs)

        if augment is False:
            train_X, train_Y = X[train], Y[train]
        else:
            train_X, train_Y = augment_training_data(X[train], Y[train], image_size, mode='image')

        Y_pred = None
        for idx, cnn_model in enumerate(cnn_models):
            reset_weights(cnn_model, initial_weights)
            cnn_model.fit(train_X, train_Y, batch_size=256, epochs=5, verbose=1, validation_data=(X[test], Y[test]))
            Y_pred_each = cnn_model.predict(X[test]).reshape(-1, 1)
            print("model {} rmse: {}".format(idx, rmse(Y[test], Y_pred_each)))
            if Y_pred is None:
                Y_pred = Y_pred_each
            else:
                Y_pred = np.concatenate((Y_pred, Y_pred_each), axis=1)
        Y_pred = np.mean(Y_pred, axis=1).reshape(-1, 1)

        if Y_pred_collection is None:
            Y_pred_collection = Y_pred
        else:
            Y_pred_collection = np.concatenate((Y_pred_collection, Y_pred), axis=1)

        print(Y_pred_collection.shape)

        avg_Y_pred = np.mean(Y_pred_collection, axis=1)
        print("t:{} h:{} rmse:{}".format(t, height_span, rmse(Y[test], Y_pred)))
        print("avg rmse:{}".format(rmse(Y[test], avg_Y_pred)))

    return avg_Y_pred, Y[test]


def output_global_average():
    with open("global_avg.csv", "w") as outputfile:
        for _ in range(2000):
            outputfile.write("{}\n".format(15.5454))


def train_full_avg_rf_model(t_span, height_span, image_size, downsample_size, learner, learner_storage_path):
    for t in t_span:
        print("training t{} h{}...".format(t, height_span))
        X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=10000)
        Y_1D = Y.reshape(-1)
        learner.fit(X, Y_1D)
        joblib.dump(learner, "{}/t{}h{}size{}.pkl".format(learner_storage_path, t, height_span, image_size))


def train_full_cnn_model(t_span, height_span, image_size, downsample_size, cnn_model, initial_weights, learner_storage_path):
    for t in t_span:
        print("training t{} h{}...".format(t, height_span))
        X, Y = load_training_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=10000)
        reset_weights(cnn_model, initial_weights)
        cnn_model.fit(X, Y, batch_size=256, epochs=10, verbose=1)
        cnn_model.save("{}/t{}h{}size{}.krs".format(learner_storage_path, t, height_span, image_size))


def load_and_test_avg_rf_model(t_span, height_span, image_size, downsample_size, learner_storage_path):
    test_y_collection = None
    for t in t_span:
        print("testing t{} h{}...".format(t, height_span))
        X = get_testA_data_sklearn(t_span=(t,), height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=2000)
        learner = joblib.load("{}/t{}h{}size{}.pkl".format(learner_storage_path, t, height_span, image_size))
        y = learner.predict(X).reshape(-1, 1)
        print(y)
        if test_y_collection is None:
            test_y_collection = y
        else:
            test_y_collection = np.concatenate((test_y_collection, y), axis=1)
    final_y = np.mean(test_y_collection, axis=1)
    print(final_y)
    with open("{}/test_y.csv".format(learner_storage_path), "w") as inputfile:
        for result in final_y:
            if result < 0:
                inputfile.write("0\n")
            else:
                inputfile.write("{}\n".format(result))


def load_and_test_cnn_model(t_span, height_span, image_size, downsample_size, learner_storage_path):
    test_y_collection = None
    for t in t_span:
        print("testing t{} h{}...".format(t, height_span))
        X = load_testA_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=2000)
        cnn_model = load_model("{}/t{}h{}size{}.krs".format(learner_storage_path, t, height_span, image_size))
        y = cnn_model.predict(X).reshape(-1, 1)
        print(y)
        if test_y_collection is None:
            test_y_collection = y
        else:
            test_y_collection = np.concatenate((test_y_collection, y), axis=1)
    final_y = np.mean(test_y_collection, axis=1)
    print(final_y)
    with open("{}/test_y.csv".format(learner_storage_path), "w") as inputfile:
        for result in final_y:
            if result < 0:
                inputfile.write("0\n")
            else:
                inputfile.write("{}\n".format(result))


def get_testA_data_sklearn(t_span, height_span, image_size, downsample_size, limit=2000):
    flatten_test_X = []
    for t in t_span:
        test_X = load_testA_data(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        if len(flatten_test_X) == 0:
            flatten_test_X = test_X.flatten().reshape(limit, -1)
        else:
            tmp_X = test_X.flatten().reshape(limit, -1)
            flatten_test_X = np.concatenate((flatten_test_X, tmp_X), axis=1)
    return flatten_test_X


def center_point_value(X):
    image_size = X.shape[-1]
    X = X.reshape(X.shape[0], X.shape[-2], X.shape[-1])
    return X[:, int(image_size / 2), int(image_size / 2)]


def round_circle(X, k):
    image_size = X.shape[-1]
    X = X.reshape(X.shape[0], X.shape[-2], X.shape[-1])
    center_X = int(image_size / 2)
    center_Y = int(image_size / 2)

    circle = np.concatenate((X[:, center_X, center_Y + k].reshape(-1, 1), X[:, center_X, center_Y - k].reshape(-1, 1),
                             X[:, center_X - k, center_Y].reshape(-1, 1), X[:, center_X + k, center_Y].reshape(-1, 1)
                            ), axis=1)
    for i in range(1, k):
        j = k - 1
        circle = np.concatenate((circle,
                                 X[:, center_X + i, center_Y + j].reshape(-1, 1),
                                 X[:, center_X - i, center_Y + j].reshape(-1, 1),
                                 X[:, center_X + i, center_Y - j].reshape(-1, 1),
                                 X[:, center_X - i, center_Y - j].reshape(-1, 1)
                                ), axis=1)

    return circle


# def handcraft_features_training(t_span, height_span, limit=10000):
#     """
#     hand craft features from data
#     for each (t, h) and downsample 1, 2, 3 do the following features,
#     center point
#     1-circle: avg, median, diff with center, variance
#     2-circle: ...
#     ...
#     5-circle: ...

#     diff features (for time variance)
#     t[i]-t[i-1]
#     center point
#     1-circle: avg-diff, median-diff
#     2-circle: ...
#     ...
#     5-circle: ...
#     """
#     image_size = 15
#     df = pd.DataFrame(np.nan, index=range(limit), columns=[])

#     # spatial features
#     for t in t_span:
#         for h in height_span:
#             for subsample in range(1, 4):
#                 X, Y = load_training_data(t, [h, ], image_size, subsample, limit)
#                 column_prefix = "t{}h{}s{}".format(t, h, subsample)
#                 center_value = center_point_value(X)
#                 df.loc[:, '{}_{}'.format(column_prefix, 'center')] = center_value
#                 for k in range(1, 6):
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'mean')] = np.mean(round_circle(X, k), axis=1)
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'mean_diff_center')] = \
#                         df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'mean')] - center_value
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'median')] = np.median(round_circle(X, k), axis=1)
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'median_diff_center')] = \
#                         df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'median')] - center_value
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix, k, 'std')] = np.std(round_circle(X, k), axis=1)

#     # temporal features:
#     for i in range(len(t_span) - 1):
#         for h in height_span:
#             for subsample in range(1, 4):
#                 t_0 = t_span[0]
#                 t_1 = t_span[i + 1]
#                 column_prefix_t0 = "t{}h{}s{}".format(t_0, h, subsample)
#                 column_prefix_t1 = "t{}h{}s{}".format(t_1, h, subsample)
#                 column_prefix_diff = "t{}-{}h{}s{}".format(t_0, t_1, h, subsample)
#                 df.loc[:, '{}_{}'.format(column_prefix_diff, 'center')] = \
#                     df.loc[:, '{}_{}'.format(column_prefix_t0, 'center')] - df.loc[:, '{}_{}'.format(column_prefix_t1, 'center')]
#                 for k in range(1, 6):
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix_diff, k, 'mean')] = \
#                         df.loc[:, '{}_k{}_{}'.format(column_prefix_t0, k, 'mean')] - df.loc[:, '{}_k{}_{}'.format(column_prefix_t1, k, 'mean')]
#                     df.loc[:, '{}_k{}_{}'.format(column_prefix_diff, k, 'median')] = \
#                         df.loc[:, '{}_k{}_{}'.format(column_prefix_t0, k, 'median')] - df.loc[:, '{}_k{}_{}'.format(column_prefix_t1, k, 'median')]

#     print(df.head())
#     df.info()

#     # temperal learner from sklearn for text
#     X_hand = np.array(df)
#     learner = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, min_child_weight=1, subsample=1, colsample_bytree=1)
#     predicted_Y = cross_val_predict(learner, X_hand, Y.reshape(-1), cv=5).reshape(-1, 1)
#     print("cv rmse: {}".format(rmse(Y, predicted_Y)))

#     return X_hand, Y

def validation_tool(t_span, height_span, image_size, downsample_size, learner, validation_method, test_ratio=0.2):
    """
    Use various learners to do cross or holdout validation
    
    ### learner:
    'rf': random forest
    'xgb': xgboost
    'cnn' or 'res'
    'rnn'
    'lstm_conv' or 'stack'

    ### validation_method
    'cross' or 'holdout'
    
    ### test_ratio
    the percentage of data is used as test data (e.g. 0.2 means 5-cross validation)
    """
    # sklearn models
    models_sklern = {
        'rf': ensemble.RandomForestRegressor(n_estimators=100, n_jobs=4),
        'xgb': xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, min_child_weight=1, subsample=1, colsample_bytree=1)
    }
    models_cnn = {
        'cnn': CnnModel,
        'res': ResModel
    }
    models_rnn_cnn = {
        'lstm_conv': ConvLstmModel,
        'stack': StackingConvRNN
    }
    models_rnn = {
        'rnn': RnnModel
    }
    validation = {
        'sklearn': {
            'cross': cross_validataion_avg_aggregate,
            'holdout': time_sensitive_validataion_avg_aggregate
        },
        'cnn': {
            'cross': cross_validataion_cnn,
            'holdout': time_sensitive_validataion_cnn            
        },
        'rnn+cnn': {
            'cross': cross_validataion_convlstm
        },
        'rnn': {
            'cross': cross_validataion_rnn
        }
    }
    if learner in models_sklern:
        test_model = models_sklern[learner]
        validation['sklearn'][validation_method](t_span, height_span, image_size, test_model,
                                                 augment=False, downsample_size=downsample_size, test_ratio=test_ratio)
    elif learner in models_cnn:
        test_model = models_cnn[learner]((image_size, image_size, len(height_span)))
        initial_weights = test_model.get_weights()
        validation['cnn'][validation_method](t_span, height_span, image_size, downsample_size,
                                             test_model, initial_weights, augment=False, test_ratio=test_ratio)
    elif learner in models_rnn_cnn:
        test_model = models_rnn_cnn[learner]((len(t_span), image_size, image_size, len(height_span)))
        validation['rnn+cnn'][validation_method](t_span, height_span, image_size, downsample_size, test_model,
                                                 test_ratio=test_ratio)

    elif learner in models_rnn:
        test_model = models_rnn[learner]((len(t_span), len(height_span) * image_size * image_size))
        initial_weights = test_model.get_weights()
        validation['rnn'][validation_method](t_span, height_span, image_size, downsample_size,
                                             test_model, initial_weights, test_ratio)


if __name__ == "__main__":
    np.random.seed(712)
    backend.set_image_data_format('channels_last')  # explicitly set the channels are in the first dimenstion

    sz_t_span = [14, 13, ]
    sz_height_span = [1, ]
    sz_image_size = 24
    sz_downsample_size = 3

    validation_tool(sz_t_span, sz_height_span, sz_image_size, sz_downsample_size, 'xgb', 'cross')

    # output trained model for test
    # train_full_avg_rf_model(t_span, height_span, image_size, downsample_size, rf, "20170410_3")
    # train_full_cnn_model(t_span, height_span, image_size, downsample_size, res_model, initial_weights, "20170411_resnet")

    # run test
    # load_and_test_avg_rf_model(t_span, height_span, image_size, downsample_size, "20170410_3")
    # load_and_test_cnn_model(t_span, height_span, image_size, downsample_size, "20170411_resnet")

    # =================== following are some validation results ==========================
    # global avg - 15.85
    # [14 13 12] [0 1] 21 rf - 14.5
    # [14 13 12] [0 1] 21 gbr downsample - 14.2
    # [14] [0 1] 21 rf - 14.0

    # ==================== fowling are neural network structures for reference ===================
    # (FAIL) input: [14 13 12 11] [1] 33*33*1 downsample-3 convLSTM (filters=64, deep_level=5, dropout=0.5)   # a larger network will be out of memory
    # (FAIL) cnn structure with 3 conv + max-pooling (maybe use even larger network later)
    # (TO_TEST) resnet structure 64
