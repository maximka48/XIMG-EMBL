#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:14:38 2021

@author: au704469
"""

import numpy as np
from maximus48 import SSIM_131119 as SSIM
from maximus48 import var
import tifffile
import matplotlib.pyplot as plt

#set the ROI of image first, the logic corresponds to FIJI (to be read (x,y,x1,y1 at the image - inverse to numpy!)
ROI = (0,0,2048,2048) 

folder = '/Users/au704469/Documents/Postdoc/Results/X-ray_tomography/Brain_organoid_P14_DESY_Nov2020/Data_test/144mm/'

image_name = 'try0_full_144mm_1_00001.tiff'

#let's read the image and crop it
image = tifffile.imread(folder+image_name)
image = image[ROI[1]:ROI[3], ROI[0]:ROI[2]]

# now let's read all flatfield files from the folder
# in our case flatfield files start with the prefix 'ff_'

#create the list all images in the folder
imlist = var.im_folder(folder)

#read ff-files
flatfield = np.asarray([tifffile.imread(folder+im) for im in imlist if im.startswith('ff_'+image_name)])
flatfield = flatfield[:,ROI[1]:ROI[3], ROI[0]:ROI[2]]

# please transpose the ff-array for the further ff-correction
flatfield = np.transpose(flatfield, (1,2,0))

# let's divide our image by the flatfield, using SSIM metrix

# images should be set as special classes:
image_class = SSIM.SSIM_const(image)
ff_class = SSIM.SSIM_const(flatfield)

# then, you can calculate SSIM metrics for each pair (data-image) - (ff-image)
index = SSIM.SSIM(image_class, ff_class).ssim()

# now, simply divide your image by flatfield-image with highest SSIM-index and get a corrected image:
result = image/flatfield[:,:,np.argmax(index)]
var.show(result[100:,100:])