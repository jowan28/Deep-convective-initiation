import numpy as np
import xarray as xr
import random
from sklearn import preprocessing, metrics
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
import sys


def main_function(domain, num_trimmed):
    # These are optimized C values for each domain size
    # found from area_optimization.py
    C_vals_land = [0.3, 0.7, 0.5, 2, 0.6]
    C_vals_sea = [4, 2, 0.8, 7, 2]
    if domain == "sea":
        C_vals = C_vals_sea
    else:
        C_vals = C_vals_land
    C = C_vals[num_trimmed]
    # import the data
    stitched = load(
        "/gws/nopw/j04/aopp/jowanf/pickled/" + domain + "/" + domain + "_dataset_10000"
    )
    Y = stitched[:, 0]
    X = stitched[:, 1:]
    # remove 'null' channels
    channel_length = 4 * 9 ** 2
    X = np.append(
        X[:, :channel_length],
        np.append(
            X[:, channel_length * 3 : channel_length * 7],
            X[:, channel_length * 9 : channel_length * 12],
            1,
        ),
        1,
    )
    # remove the exterior pixels
    X_zoomed = np.reshape(
        np.reshape(X, (len(Y), 8, 4, 9, 9))[
            :, :, :, num_trimmed : 9 - num_trimmed, num_trimmed : 9 - num_trimmed
        ],
        (len(Y), 8 * 4 * int((9 - 2 * num_trimmed) ** 2)),
    )
    # seperate and randomise dataset
    portion_for_train = 0.8
    index_rand = list(range(0, len(Y)))
    random.shuffle(index_rand)
    Y_rand = Y[index_rand]
    X_rand = X_zoomed[index_rand]
    test_limit_index = int(np.floor(len(Y) * portion_for_train))
    X_train = X_rand[:test_limit_index]
    Y_train = Y_rand[:test_limit_index]
    X_test = X_rand[test_limit_index:]
    Y_test = Y_rand[test_limit_index:]
    # normalise the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    # train the model
    print("training")
    clf = LogisticRegression(max_iter=10000, C=C, penalty="l1", solver="saga").fit(
        X_train_scaled, Y_train
    )
    # cross validate
    print("baking")
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    cv_model = cross_validate(
        clf, X_test_scaled, Y_test, cv=cv, return_estimator=True, n_jobs=2,
    )
    number_of_iterations = len(cv_model["estimator"])
    predicted = []
    accuracy_vector = []
    for logist in cv_model["estimator"]:
        predicted = logist.predict(X_test_scaled)
        accuracy_hold = metrics.accuracy_score(predicted, Y_test)
        accuracy_vector.append(accuracy_hold)
        print(accuracy_hold)
    return accuracy_vector


if __name__ == "__main__":
    domain = str(sys.argv[1])
    accuracy_mat = []
    for cut in range(5):
        accuracy = main_function(domain, cut)
        if np.shape(accuracy_mat) == np.shape([]):
            accuracy_mat = [
                accuracy,
            ]
        else:
            accuracy_mat = np.append(accuracy_mat, [accuracy,], 0)
    # save the accuracy_mat
    dump(accuracy_mat, "/gws/nopw/j04/aopp/jowanf/pickled/" + domain + "_area_cv")
