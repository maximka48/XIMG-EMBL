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

def ff_correct(distance, section):
    """the idea is to build a function that can perform ff-correction
    by giving as input the detector distance and section number"""
    
    folder = '/Users/au704469/Documents/Postdoc/Results/X-ray_tomography/Brain_organoid_P14_DESY_Nov2020/Data_test/'+distance+'mm/'

    #create the list all images in the folder
    imlist = var.im_folder(folder)

    """let's read the images and crop them - does this make sense to you? I used the same logic as for the ff-files
    we need to make the function run for each of the 3600 images for each section"""
    images = np.asarray([tifffile.imread(folder+im) for im in imlist if im.startswith('try0_full_'+distance+'mm_'+section+'_')])
    images = images[:,ROI[1]:ROI[3], ROI[0]:ROI[2]]
    
    # now let's read all flatfield files from the folder
    # in our case flatfield files start with the prefix 'ff_'

    #read ff-files
    flatfield = np.asarray([tifffile.imread(folder+im) for im in imlist if im.startswith('ff_try0_full_'+distance+'mm_'+section+'_')])
    flatfield = flatfield[:,ROI[1]:ROI[3], ROI[0]:ROI[2]]

    # please transpose the ff-array for the further ff-correction
    flatfield = np.transpose(flatfield, (1,2,0))

    # let's divide our image by the flatfield, using SSIM metrix
    """I dont know how to ake the function run for a set of images.
    We also need to divide each image by the maximum of its matrix with the ff-images"""
    
    # images should be set as special classes:
    image_class = SSIM.SSIM_const(images)
    ff_class = SSIM.SSIM_const(flatfield)

    # then, you can calculate SSIM metrics for each pair (data-image) - (ff-image)
    index = SSIM.SSIM(image_class, ff_class).ssim()

    # now, simply divide your image by flatfield-image with highest SSIM-index and get a corrected image:
    result = np.asarray(images[:]/flatfield[:,:,np.argmax(index)])
    var.show(result[100:,100:])
    """I guess it doesnt make sens to show all 3600 corrected images,
    so maybe we can print a message saying that the correction was sucessful
    after running for all the files"""