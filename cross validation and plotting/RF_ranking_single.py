from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import xarray as xr
import random
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, metrics
from joblib import load, dump
import numpy as np
import sys

def main_function(number_of_ones,clf,X_test_scaled,Y_test):
    #find the pixel ranking
    rfe = RFE(estimator=clf, n_features_to_select=number_of_ones, step=0.1)
    rfe.fit(X_test_scaled, Y_test)
    ranking = rfe.ranking_.reshape((8,36,9))
    ones_per_channel = np.sum(ranking == 1,(1,2))
    return ones_per_channel
    
    
if __name__ == "__main__":
    # These values have been found from domain optimization
    C_vals_land = 0.2
    C_vals_sea = 4
    num_ones = int(sys.argv[1])
    domain = str(sys.argv[2])
    if domain == 'sea':
        C = C_vals_sea
    else:
        C = C_vals_land

    #import the data
    stitched = load('/gws/nopw/j04/aopp/jowanf/pickled/'+domain+'/'+domain+'_dataset_10000')
    Y = stitched[:,0]
    X = stitched[:,1:]
    #remove 'null' channels
    channel_length = 4*9**2
    X = np.append(X[:,:channel_length],np.append(X[:,channel_length*3:channel_length*7],X[:,channel_length*9:channel_length*12],1),1)
    #seperate and randomise dataset
    portion_for_train = 0.8
    index_rand = list(range(0,len(Y)))
    random.Random(4).shuffle(index_rand)
    Y_rand = Y[index_rand]
    X_rand = X[index_rand]
    test_limit_index = int(np.floor(len(Y)*portion_for_train))
    X_train = X_rand[:test_limit_index]
    Y_train = Y_rand[:test_limit_index]
    X_test = X_rand[test_limit_index:]
    Y_test = Y_rand[test_limit_index:]
    #normalise the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    #train the model
    print("training")
    clf = LogisticRegression(max_iter=10000,C=C,penalty='l1',solver='saga').fit(X_train_scaled,Y_train)
    write_to_file = []
    print("ranking")
    ones_per_channel = main_function(num_ones,clf,X_test_scaled,Y_test)
    #write output to file
    file_write = '/home/users/jowanf/RF_ranking_'+domain+'.txt'
    file_object = open(file_write,'a')
    file_object.write("{}, {} = num ones\n".format(ones_per_channel,num_ones))
    file_object.close()
    