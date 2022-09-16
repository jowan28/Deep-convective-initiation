import numpy as np
import xarray as xr
import netCDF4
import glob, os
import sys
import random
from joblib import dump

#This code imports the dataset and reshapes it for use in a regression model
#This takes time, so the dataset will then be pickled for fast use elsewhere

def main_function(file_suffix,sample_num):
    #file_suffix = str(sys.argv[1])
    #file_suffix = 'sea'
    #Raw data location
    #The selected files are random
    file_root = '/gws/nopw/j04/aopp/jowanf/'
    directory = file_root + '/' + file_suffix + '/data'
    os.chdir(directory)
    pos_files = [file_root + file_suffix + '/data/' + file for file in glob.glob("*")]
    directory = file_root + '/' + file_suffix + '/negdata'
    os.chdir(directory)
    neg_files = [file_root + file_suffix + '/negdata/' + file for file in glob.glob("*")]

    random.shuffle(pos_files)
    random.shuffle(neg_files)

    #Data handling
    X = []
    Y = []
    #sample_num = int(sys.argv[2])
    num_pos = num_neg = sample_num

    for k in pos_files[:num_pos]:
        a = xr.open_dataset(k,engine="netcdf4")
        if (a.t.size == 4):
            #this exludes the extreme negative goes_imager_projection
            b = a.to_array()[:-1].values
            if (np.shape(np.where(np.isnan(b))[0])==np.shape([])):
                r=np.reshape(b,-1)
                if np.shape(X) == np.shape([]):
                    X = [r,]
                    Y = [1,]
                X = np.append(X,[r,],0)
                Y = np.append(Y,[1,],0)
    for k in neg_files[:num_neg]:
        a = xr.open_dataset(k,engine="netcdf4")
        if (a.t.size == 4):
            #this exludes the extreme negative goes_imager_projection
            b = a.to_array()[:-1].values
            if (np.shape(np.where(np.isnan(b))[0])==np.shape([])):
                r=np.reshape(b,-1)
                if np.shape(X) == np.shape([]):
                    X = [r,]
                    Y = [0,]
                X = np.append(X,[r,],0)
                Y = np.append(Y,[0,],0)

        #The Y vector will make the spine of the dataset so only one file needs to be saved
        #     Y X X X ... X X X  ^
        #     Y X X X ... X X X  |
        #     Y X X X ... X X X  |
        #     . . . . .   . . .  |
        #     . . . .  .  . . .  |
        #     . . . .   . . . .  |
        #     Y X X X ... X X X  |
        #     Y X X X ... X X X  |
        #     Y X X X ... X X X  V n_samples
        #     <--------------->
        #     n_dimensions+1=4213
        #This needs to be unstitched when opened
        
    stitched = np.append(np.transpose([Y]),X,axis=1)
    #Data pickling
    dump(stitched,'/gws/nopw/j04/aopp/jowanf/pickled/'+file_suffix+'/'+file_suffix+'_dataset_'+str(sample_num))
    return

if __name__ == "__main__":
    main_function(str(sys.argv[1]),int(sys.argv[2]))
    #argv[1]: domain: land or sea
    #argv[2]: sample number