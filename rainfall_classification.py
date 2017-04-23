"""
Predict whether the rainfall is larger than 20 or not
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics, ensemble
import xgboost as xgb
from load_rainfall import load_training_data_sklearn


def change_y_to_cat(training_y_rainfall):
    """
    change the rainfall number to category:
    1 - larger than 20
    0 - smaller than 20
    """
    threshold = np.ones(shape=(len(training_y_rainfall), 1)) * 20
    return training_y_rainfall >= threshold


def cross_validataion_classification_one_timeslot(t, height_span, image_size, downsample_size, learner, test_ratio=0.2, limit=10000, detect_outlier=False):
    """
    cross validation on one time slot
    """
    K = int(1 / test_ratio)
    Y_pred_collection = None
    X, Y = load_training_data_sklearn(t=t, height_span=height_span, image_size=image_size, downsample_size=downsample_size, limit=limit)
    Y = change_y_to_cat(Y)
    print(sum(Y))
    Y_1D = Y.reshape(-1)

    k_fold = KFold(K)
    Y_pred = np.zeros((len(Y), 1))
    for k, (train, test) in enumerate(k_fold.split(X, Y_1D)):
        learner.fit(X[train], Y_1D[train])
        Y_pred_cv = learner.predict(X[test]).reshape(-1, 1)
        Y_pred[test] = Y_pred_cv
        if detect_outlier:
            Y_pred_cv[Y_pred_cv == 1] = 0
            Y_pred_cv[Y_pred_cv == -1] = 1
        print(np.sum(Y_pred[test]))
        print("cv {} acc: {}, prec: {}, recall: {}".format(k, 
                                                           metrics.accuracy_score(Y[test], Y_pred_cv),
                                                           metrics.precision_score(Y[test], Y_pred_cv),
                                                           metrics.recall_score(Y[test], Y_pred_cv)
                                                          ))

    if Y_pred_collection is None:
        Y_pred_collection = Y_pred
    else:
        Y_pred_collection = np.concatenate((Y_pred_collection, Y_pred), axis=1)

    # avg_Y_pred = np.mean(Y_pred_collection, axis=1)
    print("t:{} h:{} acc:{}".format(t, height_span, metrics.accuracy_score(Y, Y_pred)))
    # print("avg rmse:{}".format(metrics.accuracy(Y, avg_Y_pred)))

    return Y_pred_collection, Y


if __name__ == "__main__":
    np.random.seed(712)

    sz_t = 14
    sz_height_span = [1, ]
    sz_image_size = 24
    sz_downsample_size = 3

    rf_learner = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=4)
    xgc_learner = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=50, min_child_weight=1, subsample=1, colsample_bytree=1)
    isof_learner = ensemble.IsolationForest(contamination=0.001, max_samples=0.5)

    cross_validataion_classification_one_timeslot(sz_t, sz_height_span, sz_image_size, sz_downsample_size, xgc_learner, detect_outlier=False)
