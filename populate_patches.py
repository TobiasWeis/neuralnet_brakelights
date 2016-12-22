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

maxw=80
maxh=80

# get the annotated brakelight patches
cnt = 0
ncnt = 0
for at in ats:
    if at[0].brakelight == 1:
        orig = at[2]
        img = cv2.resize(orig, (orig.shape[1]/sf, orig.shape[0]/sf))
        for t in at[1]:
            if t.position != 0: #middle brakelight
                # crop the patch so it is quadratic, that way we do not disturb when we scale to nn-input size
                w = t.bbox_lr_x/sf - t.bbox_ul_x/sf
                h = t.bbox_lr_y/sf - t.bbox_ul_y/sf

                if w > h:
                    diff = w-h

                    startx = t.bbox_ul_x/sf
                    endx = t.bbox_lr_x/sf

                    starty = t.bbox_ul_y/sf - diff/2
                    endy = t.bbox_lr_y/sf + diff/2
                elif h > w:
                    diff = h-w

                    startx = t.bbox_ul_x/sf - diff/2
                    endx = t.bbox_lr_x/sf + diff/2

                    starty = t.bbox_ul_y/sf
                    endy = t.bbox_lr_y/sf
                else:
                    startx = t.bbox_ul_x/sf
                    endx = t.bbox_lr_x/sf

                    starty = t.bbox_ul_y/sf
                    endy = t.bbox_lr_y/sf

                patch = img[
                        starty:endy,
                        startx:endx,
                        :
                        ]
                # also, try to get some negative patches of the same size (just use above/below, that should be enough)
                for ty in [-200, -150, -100, -50, 50, 100]:
                    for tx in [-100, 50, 50, 100]:
                        try:
                            negpatch = img[
                                    starty+ty:endy+ty, 
                                    startx+tx:endx+tx,
                                    :
                                    ]
                            cv2.imwrite("./brakelights/Negative/%08d.png" % (ncnt), negpatch)
                            ncnt += 1
                        except:
                            pass

                cv2.imwrite("./brakelights/Brakelight/%08d.png" % (cnt), patch)
                cnt += 1

print "Done."


