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


class Processor:
    
    def __init__(self, **kwargs):
        """Initialize parameters. 
        Normally should contain ROI, N_distances, etc
        """
        self.__dict__.update(kwargs)        
        pass
       
        
    def init_parameters(self, **kwargs):
        """Initialize parameters """
        
        self.__dict__.update(kwargs) 
        
# =============================================================================
#         params = dict()
# 
#         for key, value in kwargs.items():
#             params[key] = value
#         return params
# =============================================================================
    
    
    def init_Npad(self, compression = 8):
        """Calculate the Npad for padding
        can be adjusted with compression parameter
        By default, 8 times smaller than ROI
        """
                        
        ROI = self.ROI
        if (ROI[2]-ROI[0])>(ROI[3]-ROI[1]):
            self.Npad = (ROI[2]-ROI[0])//compression       
        else:
            self.Npad = (ROI[3]-ROI[1])//compression   
        
        #return(self.Npad)
        
        
    def init_names(self, data_name, N_distances, first_distance = 1):
        """set proper data_names"""
        
        data_names = []
        ff_names = []
        
        self.N_distances = N_distances
    
        for i in range(first_distance, N_distances + first_distance):
            data_names.append(data_name + '_' + str(i))
            ff_names.append('ff_' + data_name + '_' + str(i))
        
        return data_names, ff_names
    
    
    def init_paths(self, data_name, path, **kwargs):
        """Generate paths images & flatfields"""
        
        #set variables
        param = Processor().init_parameters(**kwargs)
        
        if 'N_distances' in self.__dict__:
            N_distances = self.N_distances
        else:
            N_distances = param['N_distances']
            
        #set data_names
        data_names, ff_names = Processor().init_names(data_name, N_distances)
        
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
        #return images, flats
        

                              
  

class F:
    
    def __init__(self, **kwargs):
        """Initialize parameters. 
        Normally should contain ROI, N_distances, etc
        """
        self.__dict__.update(kwargs)        
        pass
     
             
    def init_shared(self, dtype = 'd', **kwargs):
        '''Create shared value array for processing.'''
        
        param = Processor().init_parameters(**kwargs)
        if 'shape' not in self.__dict__:
            self.shape = param['shape']
            
        ncell = int(np.prod(self.shape))
        self.shared_array_base = Array(dtype, ncell,lock=False)
        self.dtype = dtype
        #return(shared_array_base)
    
    
    def fill_shared(self, data, **kwargs):
        '''fill shared value array with your data'''
        
        param = Processor().init_parameters(**kwargs)
        
        if 'shape' not in self.__dict__:
            try: 
                self.shape = param['shape']
            except:
                print('Error: initialize shape or type it here as shape=desired_value')
        
        if 'shared_array_base' not in self.__dict__:
            print('Error: initialize shared array first')
        
        shared_arr = F().tonumpyarray(shared_array = self.shared_array_base, 
                                      shape = self.shape,
                                      dtype = self.dtype)
        np.copyto(shared_arr, data)
        
        
    
    
    
def tonumpyarray(shared_array, shape, dtype):
    '''Create numpy array from shared memory.'''
    nparray = np.frombuffer(shared_array, dtype=dtype).reshape(shape)    
    #assert nparray.base is shared_array
    return nparray
        
        
    
    
    
    
def correct_shifts(shifts, median_dev = 5):
    """find any bad numbers which are deviate more than 5 pixels from the median
    and correct them to median of the array"""
    
    shifts = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
    
    for i in range(shifts.shape[1]):
        for j in range(shifts.shape[2]):
            shifts[:,i,j] = np.where((abs(shifts[:,i,j] - np.median(shifts[:,i,j])) > 5), np.median(shifts[:,i,j]), shifts[:,i,j])
    print('adjusted shifts') 



def rotaxis(proj, N_steps, ROI_ff = None):
    """calculate the rotation axis comparing 0 and 180 projection shift
    proj: 3D array
    N_steps: projections per degree
    N_rot: number of files that have their mirrors
    ROI_ff: where to make comparison
    
    """
    if not ROI_ff:
        ROI_ff = 0,0, proj.shape[1], proj.shape[0]
        
    cent = []
    N_rot = proj.shape[0] - 180 * N_steps
    
    for i in range(N_rot):
        distances = shift(proj[i, ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]], 
              np.flip(proj[i + N_steps*180, ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]] ,1))
        cent.append(proj[i].shape[1]/2 + distances[1]/2)
    return cent
    

        

