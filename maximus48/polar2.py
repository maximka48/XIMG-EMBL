#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:58:53 2019

@author: mpolikarpov
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2


# =============================================================================
# main function for polar interpolation
# =============================================================================

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    # horizontal direction on the image will be theta coordinate 
    #(with 0 in the image center), vertical - R coordinate (with 0 in left top corner)
    
    data = cv2.merge([data,data,data])
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_max = int(round(np.sqrt((nx//2)**2 + (ny//2)**2)))
    r_i = np.linspace(r.min(), r.max(), r_max)
    theta_i = np.linspace(theta.min(), theta.max(), 360)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    bands = []
    for band in data.T:
        zi = scipy.ndimage.map_coordinates(band, coords, order=1)
        bands.append(zi.reshape((r_max, 360)))
    output = np.dstack(bands)
    return output#, r_i, theta_i




# =============================================================================
# supplementary functions for polar interpolation
# =============================================================================

def plot_polar_image(data, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
    plt.figure()
    plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Theta Coordinate (radians)')
    plt.ylabel('R Coordinate (pixels)')
    plt.title('Image in Polar Coordinates')


def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y


def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def baseline(data1D, polynom = 5):
    '''
    substract the baseline from 2D curve
    '''
    x = np.linspace(0, data1D.shape[0], len(data1D), dtype = 'uint16')
    y = data1D
    
    pfit = np.polyfit(x, y, polynom)                         # returns polynomial coefficients
    yp = np.polyval(pfit, x)                                # fits the curve,
    smoothed = data1D - yp                                  # subtract baseline

    return smoothed


    
