#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:46:49 2018

@author: mpolikarpov
"""


import matplotlib.pyplot as plt
import cv2
import os
#from PIL import Image
import numpy as np
from numpy import mean, square, sqrt
from numpy.fft import fft2, fftshift
from joblib import Parallel, delayed



def im_folder(path):
    """
    lists images in the folder 
    
    Parameters
    __________
    path : str
    """
    
    fileformat = 'ppm','PPM','tiff','TIFF','tif','TIF','png','PNG', 'raw', 'jpg', 'JPG'
    curfol = os.getcwd()
    
    os.chdir(path)
    imfiles = os.listdir(path)
    imlist = [filename for filename in imfiles if filename.endswith(fileformat) and not (filename.startswith('.'))]
    imlist.sort()
    os.chdir(curfol)
    
    return(imlist)
    
    
    
    
    
def images_folder(path, dtype, ROI = None, data_folder_name = 'data', flatfield_folder_name = 'flatfield', asarray = True):
    
    """
    reads all images in the subfolders of the folder 'path': 

        
    Parameters
    __________
    path : str
        path to the main directory
    dtype: str
        data-type
    ROI  : tuple 
        (typically 4 numbers with (x,y,x2,y2), optional
    data_folder_name : str
        name of the folder with data, optional
    flatfield_folder_name: str
        name of the folder with flatfield, optional      
    asarray : boolean
        if True, will return an ndarray.
        Attention! to return a 3D array correctly, you need to be sure that each folder has equal number of files  
    """
    
    data=[]
    flatfield=[]
    #data_folders=[]
    #flatfield_folders=[]
    
    for root, dirs, files in os.walk(path):        
        [data.append(read_stack2(root+'/'+name, dtype, ROI, asarray = True)) for name in dirs if data_folder_name in name]
        [flatfield.append(read_stack2(root+'/'+name, dtype, ROI, asarray = True)) for name in dirs if flatfield_folder_name in name]
        #[data_folders.append(root+'/'+name) for name in dirs if data_folder_name in name]
        #[flatfield_folders.append(root+'/'+name) for name in dirs if flatfield_folder_name in name]
        
    if asarray:
        data = np.asarray(data)
        flatfield = np.asarray(flatfield)
    
    return data, flatfield
    


def images_folder2(path, data_folder_name = 'data', flatfield_folder_name = 'flatfield'):
    
    """
    finds all images in the subfolders of the folder 'path': 

        
    Parameters
    __________
    path : str
        path to the main directory
    data_folder_name : str
        name of the folder with data, optional
    flatfield_folder_name: str
        name of the folder with flatfield, optional      
    """
    
    data_folders=[]
    flatfield_folders=[]
    
    for root, dirs, files in os.walk(path):        
        [data_folders.append(root+'/'+name) for name in dirs if data_folder_name in name]
        [flatfield_folders.append(root+'/'+name) for name in dirs if flatfield_folder_name in name]
        
    
    return data_folders, flatfield_folders







def read_image(path, dtype='float32', ROI = None, opencv = False):
    
    """
    reads one image as 2D-array
    
    Parameters
    __________
    path : str
        full path to the file
    dtype: str
        data-type
    ROI: tuple
        region of interest to read, (typically 4 numbers with (x,y,x2,y2), optional
    if cv2: True
        it will use cv2 module to read files
    """
    if opencv:
        image = cv2.imread(path,cv2.IMREAD_UNCHANGED).astype(dtype)
    else:
        image = plt.imread(path).astype(dtype)
    if ROI: 
        image = image[ROI[1]:ROI[3], ROI[0]:ROI[2]]
    
    return image
    
   
    
    
def read_stack(path, dtype, ROI=None):
    """
    reads all images in the folder  with DXchange module
        advantages: fast
        disadvantages: file names should be the same, indexing starts from 0 (first file)
        
    Parameters
    __________
    path : str
        path to the file
    dtype: str
        data-type
    ROI  : tuple 
        (typically 4 numbers with (x,y,x2,y2), optional
    """
    
    import dxchange 
    
    imlist = im_folder(path)
    os.chdir(path)
    
    n = len(imlist)  
    if ROI:
        proj = dxchange.reader.read_tiff_stack(imlist[0], range(n), digit = len(str(n)), slc = ((ROI[1],ROI[3],1),(ROI[0],ROI[2],1))).astype(dtype)
    else:
        proj = dxchange.reader.read_tiff_stack(imlist[0], range(n), digit = len(str(n))).astype(dtype)
    return(proj)


    
def read_stack2(path, dtype, ROI=None, 
                asarray=True, 
                opencv = False, 
                multiprocess = False, ncore = 5):
    """
    reads all images in the folder  with pillow module: 
        advantages: files can have any names; clear logic
        disadvantages: can be slower than read_stack
        
    Parameters
    __________
    path : str
        path to the file
    dtype: str
        data-type
    ROI  : tuple 
        (typically 4 numbers with (x,y,x2,y2), optional
    asarray : boolean
        if true, returns as array, otherwise- as a list
    if cv2: True
        it will use cv2 module to read files
    """
    
    
    fileformat = 'ppm','PPM','tiff','TIFF','tif','TIF','png','PNG', 'raw'      # formats available
    imlist = [filename for filename in os.listdir(path) if filename.endswith(fileformat)]
    imlist.sort()
    
    if multiprocess:
        out = (Parallel(n_jobs=ncore, prefer='threads')(delayed(read_image)(path+i, dtype, ROI, opencv) for i in imlist))
    
    else:
        out=[]
        for i in imlist:
            out.append(read_image(path+i, dtype, ROI, opencv))
            print('reading file: ', i)
    
    
    if asarray:
        out = np.asarray(out)
        
    return out
    
    

def show(image):
    
    """
    shows the 2D image 
    
    Parameters
    __________
    image : 2D array
    """
    
    plt.figure(num=None, figsize=(20, 20), facecolor='w', edgecolor='k')
    plt.imshow(image, cmap='gray')
    return;
    
    
def fourimage(image):
    """
    does inverse fourier transform and shows the image
    
    Parameters
    __________
    data : ndarray 
        input image data 2D or 3D array
    bitnum: int 
        number of bits to save. by default is 8 bit
    """
    ft = fft2(image)
    out = np.log(abs(fftshift(ft)))
    show(out)
    return 
    
    
def imrescale(data, bitnum=16):
    """
    increases brightness/contrast and returns it as int array
    
    Parameters
    __________
    data : ndarray 
        input image data 2D or 3D array
    bitnum: int 
        number of bits to save. by default is 8 bit
    """
    bit = 2**bitnum-1
    out = (data-np.min(data))*bit/(np.max(data)-np.min(data))
    out = out.astype('uint'+str(bitnum))
    return out
    

def imrescale_interactive(data, bitnum=8):
    """
    increases brightness/contrast by user-defined value and returns it as int array
    
    Parameters
    __________
    data : ndarray 
        input image data 3D array
    bitnum: int 
        number of bits to save. by default is 8 bit
    """
    
    Flag = True
    bit = 2**bitnum-1
    
    while Flag:
        #Amp is the input from user
        Amp = float(input("Please enter amplification factor: "))
        delta = Amp * sqrt(mean(square(data))) 
        test = (data - mean(data)+delta)*bit//(2*delta)
        np.clip(test, 0, bit, test)
        show(test)
        plt.show()
        
        '''
        plt.figure(num=None, figsize=(12, 14), facecolor='w', edgecolor='k')
        plt.imshow(test, cmap='Greys_r')
        plt.show()
        '''
        #check if the user is satisfied
        flag2 = int(input("Type '1' if ok or '0' to redo the normalization:  "))
        if flag2 == 1:
            data = (data - mean(data)+delta)*bit//(2*delta)                        # dynamical range for the substraction - type 1.65 if you want to leave central 90% of signal range, type 2 for 90% and 2.6 for 99%
            np.clip(data, 0, bit, data)                                        # makes all the pixels that are bigger that 255 equal 255 and that are less 0 to be 0 (example is for 8 bit)
            data = np.array(np.round(data),dtype = 'uint'+str(bitnum))                           # Round values in array and cast as whatever-bit integer
        
            Flag = False
    return data
        
    
def make_video(stack, video_name='video_stack.avi', path=os.getcwd()):

    """
    Makes a JPG video out of the image stack
    
    Parameters
    __________
    stack : ndarray
        stack of images where the 0 axis corresponds to the image index
    video_name : str, optional
        the name of the output file
    path : str, optional
        path where the file should be saved
    """
    
    video = cv2.VideoWriter(path+'/'+video_name, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (stack.shape[1], stack.shape[2]), isColor = True)
    for i in range(stack.shape[0]):
        file = stack[i,:,:]
        #video.write(file)
        video.write(cv2.merge([file,file,file]))
    video.release()


def wavelen(energy):
    """
    Calculates the wavelength out of Energy in keV
    
    Parameters
    __________
    energy : int
    """
    
    h = 4.135667662 * 1e-18 # plank constant, keV*sec
    c = 299792458           # speed of light , m/sec
    Y = (h*c)/energy
    return Y


def maximal_intensity(image, angle = None):
    
    """
    Метод максимальной интенсивности (томо)
    
    Parameters
    __________
    image : ndarray
        stack of images where the 0 axis corresponds to the tomo slice
    angle : int
        угол обзора 

    """
    from scipy.ndimage import rotate
    if angle:
        image = rotate(image, angle, axes=(2,1))
    IM_MAX= np.max(image, axis=2)

    return IM_MAX









