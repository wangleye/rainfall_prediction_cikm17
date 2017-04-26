"""
Predict rainfall for the CIKM 2017 Competition
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics, ensemble, neighbors
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import xgboost as xgb
# import pandas as pd
from load_rainfall import load_testA_data, augment_training_data
from load_rainfall import load_training_data_sklearn, load_training_data_sklearn_4_viewpoints
# from matplotlib import pyplot
# import plotly.plotly as py
# import plotly.graph_objs as go


def rmse(Y, Y_pred):
    return np.sqrt(metrics.mean_squared_error(Y, Y_pred))


def cross_validataion_avg_aggregate(t_span, height_span, image_size, learner, augment, downsample_size, test_ratio=0.2, data_preprocess=None, limit=10000):
    K = int(1 / test_ratio)
    Y_pred_collection = None
    for t in t_span:
        X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        X, Y = preprocessing_data(X, Y, methods=data_preprocess)
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


def preprocessing_data(X, Y, methods=None):
    if methods is None:
        return X, Y

    # remove the data with missing X or too large Y (abnormal values)
    if 'remove_abnormal' in methods:
        # neg_idxs = np.unique(np.argwhere(X < 0)[:, 0])
        large_idxs = np.unique(np.argwhere(Y > 70)[:, 0])
        # merge_idx = np.unique(np.concatenate((neg_idxs, large_idxs)))
        mask = np.ones(len(Y), dtype=bool)
        mask[large_idxs] = False
        X, Y = X[mask], Y[mask]

    if 'pca' in methods:
        pca = PCA(n_components=0.95, svd_solver='full')
        pca.fit(X)
        print(pca.explained_variance_ratio_)
        X = pca.transform(X)

    if 'enhance_X' in methods:
        mean_X = np.mean(X, axis=1, keepdims=True)
        variance_X = np.var(X, axis=1, keepdims=True)
        small_X = np.count_nonzero(X < 10, axis=1).reshape(-1, 1)
        large_X = np.count_nonzero(X > 100, axis=1).reshape(-1, 1)
        X = np.concatenate((X, mean_X, variance_X, small_X, large_X), axis=1)

    return X, Y


def time_sensitive_validataion_avg_aggregate(
        t_span, height_span, image_size, learner, augment, downsample_size, test_ratio=0.2, data_preprocess=None, limit=10000):
    """
    use the first part of training data for training and the last part of the data for validation
    holdout is the percentage of the validation part
    """
    Y_pred_collection = None
    for t in t_span:
        X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        X, Y = preprocessing_data(X, Y, methods=data_preprocess)
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

    np.savetxt("result_cache.txt", np.concatenate((Y[test], avg_Y_pred, Y_pred_collection), axis=1), fmt="%.5f")

    return avg_Y_pred, Y[test]


def time_sensitive_validataion_avg_aggregate_4_viewpoints(
        t_span, height_span, image_size, learner, augment, downsample_size, test_ratio=0.2, data_preprocess=None, limit=10000):
    """
    use the first part of training data for training and the last part of the data for validation
    holdout is the percentage of the validation part
    """
    Y_pred_collection = None
    for t in t_span:
        Xs, Y = load_training_data_sklearn_4_viewpoints(
            t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
        Y_1D = Y.reshape(-1)
        num_of_recs = len(Y_1D)
        train = range(int(num_of_recs * (1 - test_ratio)))
        test = range(int(num_of_recs * (1 - test_ratio)), num_of_recs)
        Y_pred_4_viewpoints = np.zeros((len(test), 4))
        for idx, X in enumerate(Xs):
            X, _ = preprocessing_data(X, Y, methods=data_preprocess)
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


def output_global_average():
    with open("global_avg.csv", "w") as outputfile:
        for _ in range(2000):
            outputfile.write("{}\n".format(15.5454))


def train_full_avg_rf_model(t_span, height_span, image_size, downsample_size, data_preprocess, learner, learner_storage_path):
    for t in t_span:
        print("training t{} h{}...".format(t, height_span))
        X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=10000)
        X, Y = preprocessing_data(X, Y, data_preprocess)
        Y_1D = Y.reshape(-1)
        learner.fit(X, Y_1D)
        joblib.dump(learner, "{}/t{}h{}size{}.pkl".format(learner_storage_path, t, height_span, image_size))


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

def validation_tool(t_span, height_span, image_size, downsample_size, learner, validation_method, test_ratio=0.2, data_preprocess=None):
    """
    Use various learners to do cross or holdout validation
    
    ### learner:
    'rf': random forest
    'xgb': xgboost
    'knn': k nearest neighbors

    ### validation_method
    'cross' or 'holdout'
    
    ### test_ratio
    the percentage of data is used as test data (e.g. 0.2 means 5-cross validation)
    """
    # sklearn models
    models_sklern = {
        'rf': ensemble.RandomForestRegressor(n_estimators=100, n_jobs=4),
        'xgb': xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=50, min_child_weight=1, subsample=1, colsample_bytree=1),
        'knn': neighbors.KNeighborsRegressor(n_neighbors=300, weights='distance')
    }
    validation = {
        'sklearn': {
            'cross': cross_validataion_avg_aggregate,
            'holdout': time_sensitive_validataion_avg_aggregate,
            'holdout_4_view': time_sensitive_validataion_avg_aggregate_4_viewpoints
        }
    }
    if learner in models_sklern:
        test_model = models_sklern[learner]
        validation['sklearn'][validation_method](t_span, height_span, image_size, test_model,
                                                 augment=False, downsample_size=downsample_size,
                                                 test_ratio=test_ratio, data_preprocess=data_preprocess)


if __name__ == "__main__":
    np.random.seed(712)

    sz_t_span = [14, 13, 12, ]
    sz_height_span = [0, 1, ]
    sz_image_size = 21
    sz_downsample_size = 3

    validation_tool(sz_t_span, sz_height_span, sz_image_size, sz_downsample_size, 'rf', 'holdout_4_view', data_preprocess=["remove_abnormal", "enhance_X"])

    # output trained model for test
    # rf = ensemble.RandomForestRegressor(n_estimators=100)
    # train_full_avg_rf_model(sz_t_span, sz_height_span, sz_image_size, sz_downsample_size, ['remove_abnormal'], rf, "20170426_2")
    # train_full_cnn_model(t_span, height_span, image_size, downsample_size, res_model, initial_weights, "20170411_resnet")

    # run test
    # load_and_test_avg_rf_model(sz_t_span, sz_height_span, sz_image_size, sz_downsample_size, "20170426_2")
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
