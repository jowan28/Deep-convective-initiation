bonfrom sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, metrics
import xarray as xr
import random
from joblib import load, dump
import numpy as np
import pandas as pd
import sys

def main_function(domain):
    C_vals_land = 0.2
    C_vals_sea = 4
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
    #format the data
    portion_for_train = 0.8
    index_rand = list(range(0,len(Y)))
    random.shuffle(index_rand)
    Y_rand = Y[index_rand]
    X_rand = X[index_rand]
    test_limit_index = int(np.floor(len(Y)*portion_for_train))
    X_train = X_rand[:test_limit_index]
    Y_train = Y_rand[:test_limit_index]
    X_test = X_rand[test_limit_index:]
    Y_test = Y_rand[test_limit_index:]
    #scale the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_scaled = scaler.transform(X_test)
    print('training')
    clf = LogisticRegression(max_iter=10000,C=C,penalty='l1',solver='saga').fit(X_train_scaled, Y_train)
    Y_predicted = clf.predict(X_test_scaled)
    #cross validate the data
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    cv_model = cross_validate(
        clf,
        X_test_scaled,
        Y_test,
        cv=cv,
        return_estimator=True,
        n_jobs=2,
    )
    #extract the coefficients
    jit = []
    for h in range(len(cv_model["estimator"])):
        coefi = cv_model["estimator"][h].coef_[0]
        jit.append(coefi)     
    jittery_coefs = np.reshape(flatten(jit),(len(cv_model["estimator"]),len(coefi)))
    print('saving')
    #save the coefficients
    dump(jittery_coefs,'/gws/nopw/j04/aopp/jowanf/pickled/'+domain+'_jittery_coefficients')
    return
    
def flatten(l):
    return [item for sublist in l for item in sublist]

if __name__ == "__main__":
    bash_in = int(sys.argv[1])-1
    domains = ['sea','land']
    domain = domains[bash_in]
    main_function(domain)