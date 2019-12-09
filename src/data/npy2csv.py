import numpy as np
import csv
import sys
import glob
import os
import pickle

dirname = sys.argv[1]
outdir = sys.argv[2]

os.makedirs(outdir, exist_ok=True)

for filename in glob.glob(os.path.join(dirname, '*.npy')):
    data = np.load(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    outname = os.path.join(outdir, basename + ".csv")
    np.savetxt(outname, data[0], delimiter=',')
    print(outname)



