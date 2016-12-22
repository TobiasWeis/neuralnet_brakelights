#!/usr/bin/python

import os

import cv2
import numpy as np

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

import glob

from settings_brakelight import *

def extract_patches(image,s, patch_size_x, patch_size_y, stride=1, rgb=True):
    if rgb:
        X_test = np.empty((0,3, s.img_size, s.img_size),np.float32)# contains data
    else:
        X_test = np.empty((0,1, s.img_size, s.img_size),np.float32)# contains data

    #img = cv2.resize(cv2.imread(f) / 255., (s.img_size, s.img_size))

    for x in np.arange(patch_size_x/2, image.shape[1]-patch_size_x/2,stride):
        for y in np.arange(patch_size_y/2, image.shape[0]-patch_size_y/2, stride):
            patch = img[
                        y-patch_size_y/2:y+patch_size_y/2,
                        x-patch_size_x/2:x+patch_size_x/2,
                        :
                    ]
            patch = cv2.resize(patch, (s.img_size, s.img_size))

            if rgb:
                patch = patch.transpose(2,0,1).reshape(3, s.img_size, s.img_size)
                X_test = np.append(X_test, np.array([patch.astype(np.float32)]), axis=0)
            else:
                print "Not implemented yet"

    return X_test

_outfolder = "./results/"

files = glob.glob("/home/shared/data/TobisGpsSequence/gps_grabber_sqlite_36_midday_manual_cam_exposure_15_test_becker/*.png")
files.sort()

s = Settings()
net = s.net
net.load_weights_from(s.net_name)

stride = 20
sf = 1 

for i,f in enumerate(files):
    orig = cv2.imread(f)
    img = cv2.resize(orig, (orig.shape[1]/sf, orig.shape[0]/sf))
    patches = extract_patches(img,s, s.img_size, s.img_size, stride)
    print "Shape of patches from extractor: ", patches.shape
    preds = net.predict(patches)

    print "preds:"
    print preds
    print "shape:"
    print preds.shape

    img_classified = np.zeros_like(img)

    cnt = 0
    for x in np.arange(s.img_size/2, img.shape[1]-s.img_size/2, stride):
        for y in range(s.img_size/2, img.shape[0]-s.img_size/2, stride):
            try:
                if preds[cnt] == 0: # class zero is brakelight
                    img_classified[y-(s.img_size+1)/2:y+(s.img_size+1)/2,x-(s.img_size+1)/2:x+(s.img_size+1)/2] = [0,0,255]
            except Exception, e:
                print "Exception: ", e
            cnt += 1

    overlayed = 0.5 * img + 0.5 * img_classified
    big_img = np.zeros((img.shape[0], img.shape[1]*3, img.shape[2]), np.uint8)
    big_img[0:img.shape[0], 0:img.shape[1]] = img
    big_img[0:img.shape[0], img.shape[1]:img.shape[1]*2] = img_classified
    big_img[0:img.shape[0], img.shape[1]*2:img.shape[1]*3] = overlayed
    cv2.imshow("result", big_img)
    cv2.waitKey(10)
    cv2.imwrite(_outfolder + "%08d.png" % (i), big_img)

   
