#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:23:15 2019

@author: mpolikarpov
"""
import os
os.environ['OMP_NUM_THREADS'] ='1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from maximus48 import var
import numpy as np
from multiprocessing import Array
from maximus48.multiCTF2 import shift_distance as shift
 
"""
some functions here are out of classes - if you compare to the file tomo_proc.py
the reason for that - I wanted to make clases F and Processor as light as possible
 for parallel processing 
so, in a sence, I use them almost as fast dictionaries
"""

# =============================================================================
# tomo-functions
# =============================================================================

def tonumpyarray(shared_array, shape, dtype):
    '''Create numpy array from shared memory.'''
    nparray = np.frombuffer(shared_array, dtype=dtype).reshape(shape)    
    #assert nparray.base is shared_array
    return nparray
        

def correct_shifts(shifts, median_dev = 5):
    """find any bad numbers which deviate more than 5 pixels from the median
    and correct them to median of the array"""
    
    shifts = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
    
    for i in range(shifts.shape[1]):
        for j in range(shifts.shape[2]):
            shifts[:,i,j] = np.where((abs(shifts[:,i,j] - np.median(shifts[:,i,j])) > median_dev), np.median(shifts[:,i,j]), shifts[:,i,j])
    print('adjusted shifts') 


def rotaxis(proj, N_steps):
    """calculate the rotation axis comparing 0 and 180 projection shift
    proj: 3D array
    N_steps: projections per degree
    by default it compares the central part of images (between 1/4 and 3/4 of shape)
    """
    a = proj.shape[1]//4
    b = 3 * proj.shape[1]//4
    c = proj.shape[2]//4
    d = 3 * proj.shape[2]//4
        
                
    cent = []
    N_rot = proj.shape[0] - 180 * N_steps
    
    for i in range(N_rot):
        distances = shift(proj[i, a:b, c:d], np.flip(proj[i + N_steps*180, a:b, c:d] ,1))
        cent.append(proj[i].shape[1]/2 + distances[1]/2)
    
    return cent
    

def axis_raws(image1, image2, Npad = 0, RotROI = None, level = 5, window = 50):
    """Finds an axis of rotation by comparing the 1st and the 180deg image:
    image1: 2D array 
        first 2D image
    image2: 2D array
        second 2D image
    Npad: int 
        type the number here if the x axis of your image was padded
    RotROI: tuple
        ROI where to compare the images. Note that it is better to exclude some regions at the edge of the camera
        Coordinate legend:  RotROI[0] - begin line (if you look at the image, numpy logic)
                            RotROI[1] - end line
                            RotROI[2] - begin column
                            RotROI[3] - end column
    level: int 
        the number of pixels to be taken into account, please type None if you don't want to use it and want to export the whole data.
    window: int
        window is the number of pixels on the image (height) to be taken into account during comaprison
    """
    
    if not RotROI:
        RotROI = (50, image1.shape[0],
                  Npad + image1.shape[1]//8, image1.shape[1] - Npad - image1.shape[1]//8)
    

    all_cent=[]
    for i in range(RotROI[0], RotROI[1], window//2):

        im_1 = image1[i:i+window,RotROI[2]:RotROI[3]]
        im_2 = np.flip(image2[i:i+window,RotROI[2]:RotROI[3]],1)

        distances = shift(im_1, im_2)
        cent = im_1.shape[1]/2 + distances[1]/2 + RotROI[2]

        all_cent.append(cent)
        #print('center for the slice #',i,' is: ', cent)


            
    if level:
        x=[]
        y=[]

        for i in range(len(all_cent)):
            if np.absolute(all_cent[i] - np.median(all_cent)) <level:
                x.append(i * window//2)
                y.append(all_cent[i])
    
        all_cent = np.column_stack((x,y))
    
    else:
        all_cent = np.column_stack((np.linspace(0, len(all_cent), len(all_cent), dtype = 'uint16'), all_cent))
    
    #plt.plot(all_cent[:,0], all_cent[:,1], 'bo')
                
    return all_cent



def interpolate(cent, level = None):
    """interpolates the coordinates for the rotation axis with the line
     basically finds the inclination of the rotation axis through the image
   
    cent: 2D array 
        comes from axis_raws    
    level:int 
        is the number of pixels to be taken into account if you want 
        to discard all values that have more than 5 degrees difference with the median
    """  

    step = cent[1,0] - cent[0,0]
    
    if not level:
        x = cent[:,0]
        y = cent[:,1]
    
    else:
        x = []
        y = []
        for i in range(len(cent)):
            if np.absolute(cent[i,1] - np.median(cent[:,1])) < level:
                x.append(i*step)
                y.append(cent[i,1])
                
    
    pfit = np.polyfit(x, y, 1)                                                       # returns polynomial coefficients
    #yp = np.polyval(pfit, x)                                                         # fits the curve,
    
    return pfit












# =============================================================================
# #actually should be a part of the Processor class - check tomo_proc.py
# =============================================================================

def init_Npad(ROI, compression = 8):
    """Calculate the Npad for padding
    can be adjusted with compression parameter
    By default, 8 times smaller than ROI
    """
                    
    if (ROI[2]-ROI[0])>(ROI[3]-ROI[1]):
        Npad = (ROI[2]-ROI[0])//compression       
    else:
        Npad = (ROI[3]-ROI[1])//compression   
        
    return Npad 


def init_names(data_name, N_distances, first_distance):
    
    """set proper data_names"""
    print("WARNING: functions init_names and init_paths are deprecated")
    
    data_names = []
    ff_names = []
    
    if type(first_distance) == str:
        first_distance = int(first_distance)
    
    for i in range(first_distance, N_distances + first_distance):
        data_names.append(data_name + '_' + str(i))
        ff_names.append('ff_' + data_name + '_' + str(i))
        
    return data_names, ff_names 







# =============================================================================
# classes themselves
# =============================================================================
class Processor:
    __slots__ = ['ROI', 'ROI_ff', 'Npad', 'im_shape', 'images', 'flats',
                 'N_files', 'N_start']
        
    def __init__(self, ROI, folder, N_start, N_finish, compNpad = 8):
        """Initialize parameters. 
        Normally should contain ROI, N_distances, etc
        """
        self.N_start = N_start
        self.ROI = ROI
        self.N_files = (N_finish - N_start) 
        self.im_shape = (ROI[3]-ROI[1], ROI[2]-ROI[0])  
        self.Npad = init_Npad(ROI, compression = compNpad)
        
        
    def init_paths(self, data_name, path, N_distances, first_distance = 1):
        """Generate paths images & flatfields"""
    
        #set data_names
        data_names, ff_names = init_names(data_name, N_distances, first_distance = first_distance)
        
        #find images
        imlist = var.im_folder(path)
        
        #set proper paths
        images = np.zeros(N_distances, 'object') 
        flats = np.zeros(N_distances, 'object')
                
        for i in np.arange(len(images)):
            #sort image paths
            images[i] = [path+im for im in imlist if (im.startswith(data_names[i])) and not (im.startswith('.'))]
            flats[i] = [path+im for im in imlist if im.startswith(ff_names[i])]
            
        self.images = images
        self.flats = flats

                        

class F:
    __slots__ = ['shape', 'dtype', 'shared_array_base']
    
    def __init__(self, shape, dtype = 'd'):
        """Create shared value array for processing.
        """
        self.shape = shape
        self.dtype = dtype
        
        ncell = int(np.prod(self.shape))
        self.shared_array_base = Array(dtype, ncell,lock=False)       
        pass
     
        
    

