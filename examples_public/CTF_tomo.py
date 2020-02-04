#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:34:02 2019
This is an example of a parallelised script that does:
- Flatfield correction with SSIM
- 4-distance CTF-reconstruction (incl. correction for shifts)
- Stripe artifact reduction
- Automatic rotation axis detection
- Tomo-reconstruction (paralle-beam case)
- save to h5 (big data viewer format)
- calculate inclination of the rotation axis (optional)

@author: mpolikarpov
"""
import os
os.environ['OMP_NUM_THREADS'] ='1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
#os.system('taskset -cp 0-100 %d' % os.getpid())

import sys, time
import dxchange, tomopy
import numpy as np
import tifffile
from contextlib import closing
from multiprocessing import Pool
import gc

from maximus48 import var
from maximus48 import SSIM_131119 as SSIM_sf 
from maximus48 import multiCTF2 as multiCTF
from maximus48.SSIM_131119 import SSIM_const 
from maximus48.tomo_proc2 import Processor, F, correct_shifts, rotaxis, tonumpyarray, axis_raws, interpolate

from scipy.ndimage import rotate
from pybdv import make_bdv 



# =============================================================================
#           parameters for phase retrieval with CTF
# =============================================================================
N_steps = 10                                                                   # Number of projections per degree
N_start = 1                                                                    # index of the first file
N_finish = 3600                                                                # index of the last file

N_distances  = 4                                                               # number of distance in phase-retrieval
pixel = 0.1625 * 1e-6                                                          # pixel size 
distance = np.array((6.25, 6.75, 7.35, 8.25), dtype = 'float32') * 1e-2                # distances of your measurements 
energy = 18                                                                    # photon energy in keV
beta_delta = 0.1
zero_compensation = 0.1
#inclination = 0.1                                                             # inclination of the rotation axis

ROI = (0,20,2048,2048)                                                         # ROI of the image to be read (x,y,x1,y1 at the image - inverse to numpy!)
cp_count = 85                                                                  # number of cores for multiprocessing



#data_name = sys.argv[1]
#folder = sys.argv[2]
#folder_result = sys.argv[3]

data_name = 'DATA_NAME'
folder = 'full_path_To_Data'
folder_result = 'full_Path_to_save'





# =============================================================================
#    prepartion work 
# =============================================================================
print(data_name, "started with %d cpus on" % cp_count, time.ctime())
time1 = time.time()

#calculate parameters for phase-retrieval
wavelength = var.wavelen(energy)
fresnelN = pixel**2/(wavelength*distance)

#create save folder if it doesn't exist
if not os.path.exists(folder_result):
    os.makedirs(folder_result)


# create a class to store all necessary parameters for parallelization
Pro = Processor(ROI, folder, N_start, N_finish, compNpad = 8)                 

#set proper paths
Pro.init_paths(data_name, folder, N_distances) 

#allocate memory to store flatfield
shape_ff = (N_distances, len(Pro.flats[0]), Pro.im_shape[0], Pro.im_shape[1]) 
ff_shared = F(shape = shape_ff, dtype = 'd')

#read ff-files to memory
ff = tonumpyarray(ff_shared.shared_array_base, ff_shared.shape, ff_shared.dtype)
for i in range(N_distances):
    ff[i] = tifffile.imread(Pro.flats[i])[:,ROI[1]:ROI[3], ROI[0]:ROI[2]] 

#calculate ff-related constants
Pro.ROI_ff = (ff.shape[3]//4, ff.shape[2]//4,3 * ff.shape[3]//4, 3 * ff.shape[2]//4)    # make ROI for further flatfield and shift corrections, same logic as for normal ROI
ff_con = np.zeros(N_distances, 'object')                                                # array of classes to store flatfield-related constants
for i in np.arange(N_distances):    
    ff_con[i] = SSIM_const(ff[i][:,Pro.ROI_ff[1]:Pro.ROI_ff[3], 
                                   Pro.ROI_ff[0]:Pro.ROI_ff[2]].transpose(1,2,0))


#allocate memory to store ff-indexes
indexes = F(shape = (N_finish - N_start, N_distances), dtype = 'i' )

#allocate memory to store shifts
shifts = F(shape = (N_finish - N_start, N_distances, 2), dtype = 'd')

#allocate memory to store filtered files
proj = F(shape = (Pro.N_files, shape_ff[2], shape_ff[3] + 2*Pro.Npad), dtype = 'd' )

print('finished calculation of ff-constants and memory allocation in ', time.time()-time1)







# =============================================================================
# =============================================================================
# # Processing module
# =============================================================================
# =============================================================================

# =============================================================================
# functions for parallel processing 
# =============================================================================

def init():
    global Pro, ff_shared, ff_con, shifts, indexes

def init2():
    global Pro, ff_shared, proj, shifts, indexes
    

def read_flat(j):    
    """
    j: int
        an index of the file that should be processed 
    Please note, j always starts from zero
    To open correct file, images array uses images[i][j + N_start-1]
    """
    
    #global ff_shared, ff_con, shifts, indexes, Pro
    
    #set local variables
    ff = tonumpyarray(ff_shared.shared_array_base, ff_shared.shape, ff_shared.dtype)
    shift = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
    LoI = tonumpyarray(indexes.shared_array_base, indexes.shape, indexes.dtype)
    
    ROI_ff = Pro.ROI_ff
    ROI = Pro.ROI
    images = Pro.images
    N_start = Pro.N_start
    
    #read image and do ff-retrieval    
    filt = []
    for i in np.arange(len(images)):
        im = tifffile.imread(images[i][j + N_start-1])[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        
        index = SSIM_sf.SSIM(SSIM_const(im[ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]]), 
                                        ff_con[i]).ssim()
        
        im = im/ff[i][np.argmax(index)]
        filt.append(im)
        LoI[j,i] = np.argmax(index)


    #calculate shift for holograms
    for i in np.arange(len(filt)):
        shift[j,i] = (multiCTF.shift_distance(filt[0][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]], 
                                              filt[i][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]], 
                                             100))
# =============================================================================
#     #calculate shift for retrieved images
#     for i in np.arange(len(filt)):
#         shift[j,i] = (multiCTF.find_shift_CTF(filt[0][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]], 
#                                              filt[i][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]], 
#                                              beta_delta, fresnelN[0], fresnelN[i], 
#                                              accuracy = 100, zero_compensation = zero_compensation))
# =============================================================================
    #print('sucessfully processed file: ', images[0][j + N_start-1])



def shift_retrieve(j):
    """
    j: int
        an index of the file that should be processed 
    """
     
    #set local variables
    ff = tonumpyarray(ff_shared.shared_array_base, ff_shared.shape, ff_shared.dtype)
    proj_loc = tonumpyarray(proj.shared_array_base, proj.shape, proj.dtype)
    shift = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
    LoI = tonumpyarray(indexes.shared_array_base, indexes.shape, indexes.dtype)
    
    
    ROI = Pro.ROI
    images = Pro.images
    Npad = Pro.Npad
    N_start = Pro.N_start
    
    
    #read image and do ff-retrieval    
    filt = []                                                           
    for i in np.arange(len(images)):
        im = tifffile.imread(images[i][j + N_start-1])[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        im = im/ff[i][LoI[j,i]]
        filt.append(im)
    
    #shift images
    filt = multiCTF.shift_imageset(np.asarray(filt), shift[j])
    filt = np.asarray(filt)

    #do CTF retrieval 
    filt = np.pad(filt, ((0,0),(Npad, Npad),(Npad, Npad)), 'edge')               # padding with border values
    filt = multiCTF.multi_distance_CTF(filt, beta_delta, 
                                           fresnelN, zero_compensation)
    filt = filt[Npad:(filt.shape[0]-Npad),:]                                     # unpad images from the top
    
    #rotate the image to compensate for the inclined rotation axis
    #filt = rotate(filt, -inclination, mode = 'nearest')
    
    #save to memory
    proj_loc[j] = filt    
    #print('sucessfully processed file: ', images[0][j + N_start-1])
    
    




# =============================================================================
# Process projections
# =============================================================================
    
#calculate ff-indexes and shifts
time1 = time.time()
with closing(Pool(cp_count, initializer = init)) as pool:    
    pool.map(read_flat, np.arange(Pro.N_files))
print('time for ff+shifts: ', time.time()-time1)


# correct shift for anomalous values
correct_shifts(shifts, median_dev = 10)

#shift images and do phase retrieval
time1 = time.time()
with closing(Pool(cp_count, initializer = init2)) as pool:    
    pool.map(shift_retrieve, np.arange(Pro.N_files))               
pool.close()
pool.join()
print('time for make shifts and phase retrieval: ', time.time()-time1)

proj = tonumpyarray(proj.shared_array_base, proj.shape, proj.dtype)

#remove vertical stripes with wavelet-fourier filtering
time1 = time.time()            
proj = tomopy.prep.stripe.remove_stripe_fw(proj,level=3, wname=u'db25', sigma=2, pad = False)
print('time for stripe removal ', time.time()-time1)

#np.save(folder_result + data_name +  '_proj.npy', proj)

# save what you need and release memory
#shifts_2_save = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
#np.save(folder_result + data_name +  '_proj.npy', proj)
#np.save(folder_result + data_name +  '_shifts.npy', shifts_2_save)
ff_con = None
ff_shared = None
indexes = None
shifts = None
gc.collect()





# =============================================================================
#  tomo reconstruction and save
# =============================================================================
#find rotation center from phase-retrieved images
cent = rotaxis(proj, N_steps)
cent = np.median(cent)
#print('done with rotation axis, X coordinate = ', cent)

n = proj.shape[0]
angle = np.pi*np.arange(n)/(N_steps*180)

time1 = time.time()
recon = tomopy.recon(proj, angle, center = cent, algorithm = 'gridrec', filter_name = 'shepp')
print('time for tomo_recon ', time.time()-time1)

#np.save(folder_result + data_name + 'save_tomo_result.npy', recon)
outs = var.imrescale(recon[:,Pro.Npad : recon.shape[1]- Pro.Npad,
                             Pro.Npad : recon.shape[2]- Pro.Npad], 16)
#crop additionally
outs = outs[:,500:1500, 150:1900]
    
folder_tiff = folder_result + 'plane_tiff/'
dxchange.write_tiff_stack(outs, fname= folder_tiff + data_name + '_tomo/tomo')
#np.save(folder_result + 'tomo.npy', outs)



# =============================================================================
# save all parameters to the txt file
# =============================================================================
folder_param = folder_result + 'parameters/' 
os.mkdir(folder_param) 
os.mknod(folder_param + data_name + '_parameters.txt')
with open(folder_param + data_name + '_parameters.txt', 'w') as f:
    f.write(time.ctime() + '\n')
    f.write("data_path = %s\n" % folder)
    f.write("ROI =  %s\n" % str(ROI))
    f.write("pixel size = %s\n" %str(pixel))
    f.write("distances = %s\n" %str(distance))
    f.write("energy = %s\n" %str(energy))
    f.write("beta_delta = %s\n" %str(beta_delta))
    f.write("fresnel Number = %s\n" %str(fresnelN))
    f.write("zero_compensation = %s\n" %str(zero_compensation))
    f.write("Npad = %s\n" %str(Pro.Npad))
    f.write("center of rotation = %s\n" %str(cent))
    #f.write("inclination of rotation axis (degrees) = %s\n" %str(inclination))
    f.write("projections per degree = %s\n" %str(N_steps))
    
        
    
# =============================================================================
# save as h5    
# =============================================================================
#cast
data = outs
data -= data.min()
data = (data/data.max())* 32767 
data = data.astype('int16')

# set the factors for downscaling, for example 2 times isotropic downsampling by a factor of 2
scale_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] 

# set the downsampling mode, 'mean' is good for image data, for binary data or labels
# use 'nearest' instead
mode = 'interpolate'

# resolution of the data, set appropriately
resolution = [pixel*1e6 , pixel*1e6 , pixel*1e6] 

#save big data format
folder_h5 = folder_result + 'bdv/'
os.mkdir(folder_h5)
make_bdv(data, folder_h5 + data_name, downscale_factors=scale_factors,
                 downscale_mode=mode, resolution=resolution,
                 unit='micrometer', setup_name = data_name) 





















# =============================================================================
# # =============================================================================
# # find inclination of rotation axis 
# # =============================================================================
# image1 = proj[0]
# image2 = proj[1800]
# 
# #find rotation centers
# centersX =  axis_raws(image1, image2, Npad=256)
# #plt.plot(centersX[:,0], centersX[:,1], 'bo')
# inclination = interpolate(centersX)                          # this is the inclination (polinomial function) by which the tomogram is rotated
# inclination = np.arctan(inclination[0]) * (360/(2*np.pi))                       # this is the inclination (angle in degrees) by which the tomogram is rotated
# =============================================================================


# =============================================================================
# # =============================================================================
# # find inclination of rotation axis - old
# # =============================================================================
# time1 = time.time()
# 
# image1 = var.read_image(images[0][0], 'uint16', ROI, True)
# image2 = var.read_image(images[0][180*N_steps], 'uint16', ROI, True)
# 
# #correct for flatfield to make it better
# image1 = SSIM_sf.divide(image1, ff[0], uy[0], deltaY[0], sy[0])
# image2 = SSIM_sf.divide(image2, ff[0], uy[0], deltaY[0], sy[0])
# 
# #find rotation centers
# centersX =  rotaxis.rotation().axis_raws(image1, image2, Npad=0)
# #plt.plot(centersX[:,0], centersX[:,1], 'bo')
# inclination = rotaxis.rotation().interpolate(centersX)                          # this is the inclination (polinomial function) by which the tomogram is rotated
# inclination = np.arctan(inclination[0]) * (360/(2*np.pi))                       # this is the inclination (angle in degrees) by which the tomogram is rotated
# 
# print('time to find the inclination: ', time.time()-time1)
# =============================================================================



