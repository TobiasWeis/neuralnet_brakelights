#!/usr/bin/python

import sys
sys.path.append('../brakelight/dataset/')
sys.path.append('../brakelight/scripts/')
from common import DBInterface
import cv2
import time
import numpy as np
##import matplotlib.pyplot as plt
import math
import time
import matplotlib.pyplot as plt
import pandas as pd

from loader_annots_tls_imgs import *

print "Getting annotations and images"
st = time.time()
#ats = create_annot_taillight_img_dict(seq_id=47)
ats = create_annot_taillight_img_dict()
print "Done. Took ", time.time() - st

sf = 1

maxw=32
maxh=32
minsize = 8

# get the annotated brakelight patches
cnt = 0
ncnt = 0
for at in ats:
    if at[0].brakelight == 1 or at[0].brakelight == 0:
        orig = at[2]
        img = cv2.resize(orig, (orig.shape[1]/sf, orig.shape[0]/sf))
        mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

        for t in at[1]:
            if at[0].brakelight == 0 and t.taillight == 0:
                outdir = "OFF"
            elif at[0].brakelight == 1:
                outdir = "BL"
            elif at[0].brakelight == 0 and t.taillight == 1:
                outdir = "TL"
            else:
                outdir = "WTF"

            if t.position != 0: #middle brakelight
                # get the original width/height of the annotation
                if t.bbox_ul_x % 2 != 0:
                    t.bbox_ul_x -= 1
                if t.bbox_lr_x % 2 != 0:
                    t.bbox_lr_x += 1
                if t.bbox_ul_y %2 != 0:
                    t.bbox_ul_y -= 1
                if t.bbox_lr_y % 2 != 0:
                    t.bbox_lr_y += 1

                w = t.bbox_lr_x/sf - t.bbox_ul_x/sf
                h = t.bbox_lr_y/sf - t.bbox_ul_y/sf

                # if the brakelight is larger than our desired size, skip it
                if (w > 32 or h > 32) or (w < minsize and h < minsize):
                    continue

                # calculate the diff to the desired width/height
                diff_w = maxw-w
                diff_h = maxh-h

                startx = t.bbox_ul_x-diff_w/2
                endx = t.bbox_lr_x+diff_w/2

                starty = t.bbox_ul_y-diff_h/2
                endy = t.bbox_lr_y+diff_h/2

                mask[starty:endy, startx:endx] = 255
                patch = img[
                        starty:endy,
                        startx:endx,
                        :
                        ]
                # also, try to get some negative patches of the same size (just use above/below, that should be enough)


                # randomly draw negative patch, make sure it does not intersect
                # with the brakelight


                cv2.imwrite("./brakelights/%s/%08d.png" % (outdir, cnt), patch)
                cnt += 1

        # randomly select position in image
        # check if it intersects with the mask
        # if so: select random patch again
        while True:
            randx = np.random.randint(maxw/2, img.shape[1]-maxw/2)
            randy = np.random.randint(maxh/2, img.shape[0]-maxh/2)
            if np.sum(mask[randy-maxh/2:randy+maxh/2,randx-maxw/2:randx+maxw/2]) == 0:
                negpatch = img[randy-maxh/2:randy+maxh/2,randx-maxw/2:randx+maxw/2]
                cv2.imwrite("./brakelights/NEG/%08d.png" % (ncnt), negpatch)
                ncnt += 1
                break


print "Done."


