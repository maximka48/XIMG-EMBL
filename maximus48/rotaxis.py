#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:58:53 2019

@author: mpolikarpov

Замечанания от 171019
На примере образца номер 5_3 из данных Бенедикта (Бремен) с эксперимента на P14_EMBL от 220919
если не восст фазу то центр вращения 872.475 и 872.135 (старый алгоритм - когда просто сравниваются две картинки как целое)
если есть восст фазы то центр 1384.5525 и 1384.505 (старый алгоритм) - в пересчете это 872.5525 и 872.5
выввод - обрезание картинки вообще говоря уменьшает разброс между старым и новым алогоритмом (наверное логично так как уменьшается шум)

На примере образца 5_2 
я увидел что восттановление фазы вносит бОльшую погрешность в определение центра вращения
так что лучше находить ось вращения на RAW изображениях но с флетфилд-коррекцией
если не восст фазу то центр вращения 870.017 и 869.83 (старый алгоритм)



Общий вывод - находи ось на картинках с flatfield-correction но без восстановления фазы
Края картинок можешь чуть обрезать
"""


import numpy as np
import matplotlib.pyplot as plt
import os
from maximus48.multiCTF2 import shift_distance as shift
from multiprocessing import Pool 








class rotation:
    '''
    description
    '''
    
# =============================================================================
#     def __init__(self, data, origin = None):
#         """Constructor"""
#         self.data = data
#         self.origin = origin
# =============================================================================
    def __init__(self):
        """Constructor"""
        pass
    
    
    
    

    # =============================================================================
    # some functions to effectively fint the rotation axis in the tomography experiment
    # =============================================================================
    
    
    def axis_raws(self, image1, image2, Npad = 0, RotROI = None, level = 5, window = 50):
        """Finds an axis of rotation by comparing the 1st and the 180deg image:
        image1: 2D array 
            first 2D image
        image2: 2D array
            second 2D image
        Npad: int 
            type the number here if the x axis of your image was padded
        RotROI: tuple
            ROI where to compare the images. Note that it is better to exclude some regions at the edge of the camera
        level: int 
            the number of pixels to be taken into account, please type None if you don't want to use it and want to export the whole data.
        window: int
            window is the number of pixels on the image (height) to be taken into account during comaprison
        """
        
        if not RotROI:
            RotROI = (0, image1.shape[0],
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
    
    




    def interpolate(self, cent, level = None):
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
#     def axis_raws_parallel(self, image1, image2, Npad = 200, RotROI = None, level = 5, ncore = 10):
#         """Finds an axis of rotation by comparing the 1st and the 180deg image:
#         image1 - first 2D image
#         image2 - second 2D image
#         Npad - type the number here if the x axis of your image was padded
#         RotROI - ROI where to compare the images. Note that it is better to exclude some regions at the edge of the camera
#         level - is the number of pixels to be taken into account, please type None if you don't want to use it and want to export the whole data.
#             """
#         
#         if not RotROI:
#             RotROI = (0, image1.shape[0],
#                       Npad+image1.shape[1]//8, image1.shape[1]-Npad-image1.shape[1]//8)
#         
#     
#         all_cent=[]
# 
#         im_1 = image1[:,RotROI[2]:RotROI[3]]
#         im_2 = np.flip(image2[:,RotROI[2]:RotROI[3]],1)
#         
# # =============================================================================
# #         def process(i):
# #     
# #             distances = shift(im_1[i:i+1], im_2[i:i+1])
# #             cent = im_1.shape[1]/2 + distances[1]/2 + RotROI[2]
# #     
# #             return cent
# # =============================================================================
# 
# 
#         os.environ['OMP_NUM_THREADS'] ='1'
#         with Pool(ncore) as pool:    
#             all_cent = pool.map(process, np.arange(RotROI[0], RotROI[1]))
#         os.environ['OMP_NUM_THREADS'] ='100'
# 
# 
# 
#     def center(i):
#         distances = shift(im_1[i:i+1], im_2[i:i+1])
#         cent = im_1.shape[1]/2 + distances[1]/2
#         
#         return cent
#         
#         
#                 
# # =============================================================================
# #         if level:
# #             x=[]
# #             y=[]
# # 
# #             for i in range(len(all_cent)):
# #                 if np.absolute(all_cent[i] - np.median(all_cent)) <level:
# #                     x.append(i)
# #                     y.append(all_cent[i])
# #         
# #             all_cent = np.column_stack((x,y))
# #         
# #         else:
# #             all_cent = np.column_stack((np.linspace(0, len(all_cent), len(all_cent), dtype = 'uint16'), all_cent))
# # =============================================================================
#         
#         #plt.plot(all_cent[:,0], all_cent[:,1], 'bo')
#                     
#         return all_cent
# =============================================================================

        
        
        
        
        
        
    
    
    
    
    
    
    
    