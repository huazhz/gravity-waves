import numpy as np
import glob
import os
import h5py as h5

with h5.File('train70.h5', 'w') as h5f:
    for f in glob.glob('train_data/*.dat'):
        print f
        data = np.loadtxt(f)
        name = f[:-4]
        h5f[name] = data
