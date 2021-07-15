import h5py
import re
import math
import time
import glob
import numpy as np
import pandas as pd
import os
import sys

def main():
    # Get the name of the target_name where the files will be stored
    target_name= sys.argv[1]

    # Make the new target_name and the subfolders
    os.mkdir(str(target_name))
    os.mkdir(str(target_name)+'/train_data_fixed')
    os.mkdir(str(target_name)+'/validation_data_fixed')
    os.mkdir(str(target_name)+'/test_data_fixed')

    # Convert all files
    print("Converting train data")
    names=glob.glob("data_norm/train_data_fixed/*.npz")
    for a in range(len(names)):
        npzfile = np.load(names[a])
        x1, x2, y = npzfile['X1'], npzfile['X2'], npzfile['y']
        f = h5py.File(str(target_name)+"/train_data_fixed/train_dataset_"+str(a)+".hdf5", "w")
        f.create_dataset('X1', data=npzfile['X1'])
        f.create_dataset('X2', data=npzfile['X2'])
        f.create_dataset('y', data=npzfile['y'])
        f.close()

    print("Converting test data")
    names=glob.glob("data_norm/test_data_fixed/*.npz")
    for a in range(len(names)):
        npzfile = np.load(names[a])
        x1, x2, y = npzfile['X1'], npzfile['X2'], npzfile['y']
        f = h5py.File(str(target_name)+"/test_data_fixed/test_dataset_"+str(a)+".hdf5", "w")
        f.create_dataset('X1', data=npzfile['X1'])
        f.create_dataset('X2', data=npzfile['X2'])
        f.create_dataset('y', data=npzfile['y'])
        f.close()

    print("Converting validation data")
    names=glob.glob("data_norm/validation_data_fixed/*.npz")
    for a in range(len(names)):
        npzfile = np.load(names[a])
        x1, x2, y = npzfile['X1'], npzfile['X2'], npzfile['y']
        f = h5py.File(str(target_name)+"/validation_data_fixed/validation_dataset_"+str(a)+".hdf5", "w")
        f.create_dataset('X1', data=npzfile['X1'])
        f.create_dataset('X2', data=npzfile['X2'])
        f.create_dataset('y', data=npzfile['y'])
        f.close()

if __name__ == "__main__":
     main()
