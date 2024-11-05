import cv2
import datetime as dt
import h5py
import numpy as np
import os
import fnmatch

synthText_result_path = "SynthText/results/SynthText.h5"

img_path = "SynthText/data/dset.h5"


with h5py.File(img_path, 'r') as db:
    
    print(db)
    
    print(db.keys())
    print(db['image'].keys())
    
    
    
    print("hello")