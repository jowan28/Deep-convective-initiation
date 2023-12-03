import numpy as np
import xarray as xr
import random
from sklearn import preprocessing, metrics
from joblib import load, dump
from sklearn.linear_model import LogisticRegression
import sys


def main_function(num_trimmed, c_num, domain):

    C_values = flatten(
        [
            np.arange(0.01, 0.1, 0.01),
            np.arange(0.2, 1, 0.1),
            np.arange(2, 15, 1),
            np.arange(20, 50, 10),
        ]
    )
    C_val = C_values[c_num]

    stitched = load(
        "/gws/nopw/j04/aopp/jowanf/pickled/" + domain + "/" + domain + "_dataset_10000"
    )
    Y = stitched[:, 0]
    X = stitched[:, 1:]
    # remove null channels
    channel_length = 324
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

    portion_for_train = 0.8
    index_rand = list(range(0, len(Y)))
    random.Random(4).shuffle(index_rand)
    Y_rand = Y[index_rand]
    X_rand = X_zoomed[index_rand]
    test_limit_index = int(np.floor(len(Y) * portion_for_train))
    X_train = X_rand[:test_limit_index]
    Y_train = Y_rand[:test_limit_index]
    X_test = X_rand[test_limit_index:]
    Y_test = Y_rand[test_limit_index:]

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    print("training")
    clf = LogisticRegression(max_iter=10000, C=C_val, penalty="l1", solver="saga").fit(
        X_train_scaled, Y_train
    )
    Y_predicted = clf.predict(X_test_scaled)

    accuracy = metrics.accuracy_score(Y_test, Y_predicted)
    f1_score = metrics.f1_score(Y_test, Y_predicted)

    # write output to file
    file_write = (
        "/home/users/jowanf/area_optimize_" + domain + "_" + str(num_trimmed) + ".txt"
    )
    file_object = open(file_write, "a")
    file_object.write(
        "{} = accuracy, {} = F-score, C = {}\n".format(accuracy, f1_score, C_val)
    )
    file_object.close()
    return


def flatten(l):
    return [item for sublist in l for item in sublist]


if __name__ == "__main__":
    bash_input = int(sys.argv[1]) - 1
    number_trimmed = np.floor(bash_input / 33)
    c_num = bash_input % 33
    domain = str(sys.argv[2])
    main_function(int(number_trimmed), c_num, domain)
