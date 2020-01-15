"""
Created on Fri Oct  12 15:46:49 2018

@author: mpolikarpov
"""


from numpy import floor, meshgrid, multiply, divide, square, sin, cos, sqrt, mean, pi, fliplr, flipud, rot90, ceil, arange, concatenate
import numpy
from numpy.fft import ifftshift, fft2, ifft2
from scipy.special import erfc
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from maximus48.monochromaticCTF import single_distance_CTF as sdCTF








def shift_distance(image1, image2, accuracy = 100):
    """
    Finds lateral shift between two images 
    
    Parameters
    __________
    image1 : 2D array

    image2 : 2D array
        y axis
    accuracy: int
        Upsampling factor. Images will be registered within 1 / upsample_factor of a pixel. For example upsample_factor == 20 means the images will be registered within 1/20th of a pixel.    
    """
    
    shift, error, diffphase = register_translation(image1, image2, 100)
    
    return shift



def find_shift_CTF(image1, image2, beta_delta, fresnelNumber1, fresnelNumber2, accuracy=100, zero_compensation=0.01):
    """
    Finds the lateral shift between two holographic projections by reconstructing the phase
    
    Parameters
    __________
    image1 : 2D array

    image2 : 2D array
        y axis
    accuracy: int
        Upsampling factor. Images will be registered to within 1 / upsample_factor of a pixel. For example upsample_factor == 20 means the images will be registered within 1/20th of a pixel.    
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel Number(1,2): float 
        fresnel numbers for corresponding pictures
        can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    zero_compensation: float
        just a parameter, should be small, for ex. 0.01
    """
        
    image1_CTF = sdCTF(image1, beta_delta, fresnelNumber1, zero_compensation) 
    image2_CTF = sdCTF(image2, beta_delta, fresnelNumber2, zero_compensation) 
        
    shift = shift_distance(image1_CTF, image2_CTF, accuracy)
    
    return shift



def shift_image(image, shift):
    """
    Laterally shifts one image. 
    
    Parameters
    __________
    image : 2D array
    shift: float
        shift value
    """
    
    back_image = fourier_shift(numpy.fft.fftn(image), shift)
    back_image = numpy.fft.ifftn(back_image).real
    return back_image



def shift_imageset(image, shift):
    """
    Aligns every image in the image-set by lateral shift. 
    
    Parameters
    __________
    image : 3D array
        first axis is images acquired at different distances
    shift: 3D aray 
        shift values for every image
    """
    #out = numpy.zeros(image.shape, dtype = image.dtype)
    out = []
    for i in numpy.arange(len(image)):
        out.append(shift_image(image[i], shift[i]))
    
    return out #numpy.asarray(out)


# =============================================================================
# def shift_imageset(image, shift):
#     """
#     Aligns every image in the image-set by lateral shift. 
#     
#     Parameters
#     __________
#     image : 3D array
#         first axis is images acquired at different distances
#     shift: 3D aray 
#         shift values for every image
#     """
#     out = numpy.zeros(image.shape, dtype = image.dtype)
#     for i in numpy.arange(len(image)):
#         out[i] = shift_image(image[i], shift[i])
#     
#     return out
# 
# =============================================================================


# =============================================================================
# 
# 
# def mdCTF_imageset(image, beta_delta, fresnelN, accuracy, zero_compensation, Npad = 200):
#     """
#     Shifts images and returns result of multi-distance CTF-retrieval
#     
#     Parameters
#     __________
#     image : 3D array
#         first axis is images acquired at different distances
#     beta_delta: int
#         absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
#     fresnel Number(1,2): array 
#         fresnel numbers for corresponding pictures
#         can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
#     accuracy: int
#         Upsampling factor. Images will be registered to within 1 / upsample_factor of a pixel. For example upsample_factor == 20 means the images will be registered within 1/20th of a pixel.    
#     beta_delta: int
#         absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
#     fresnelN: tuple 
#         fresnel numbers for corresponding pictures, can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
#     zero_compensation: float
#         just a parameter, should be small, for ex. 0.01
#     Npad : int
#         edge padding before phase-retrieval
#     """
#     
#     out = numpy.zeros(image.shape, dtype = image.dtype)
#     
#     #shift images
#     for i in numpy.arange(out.shape[0]):
#         out[i] = shift_image_CTF(image[0], image[i], beta_delta, fresnelN[0], fresnelN[i], accuracy, zero_compensation)
#     
#     
#     
#     
#     
#     #pad images
#     out = numpy.pad(out, ((0,0),(Npad, Npad),(Npad, Npad)), 'edge')        
#     # multiple-distance CTF-retrieval
#     out = multi_distance_CTF(out, beta_delta, fresnelN, zero_compensation)
#     #unpad images
#     out = out[Npad:(out.shape[0]-Npad),Npad:(out.shape[1]-Npad)]
#     
#     return out
# =============================================================================
















def multi_distance_CTF(image, beta_delta, fresnelNumber, zero_compensation):
    """
    This function makes the CTF phase-retrieval from the multiple-distance images.
    Please refer to the book of Tim Salditt "Biomedical imaging", 2017, ISBN = 978-3-11-042668-7, ISBN (PDF)= 978-3-11-0426694
    equation 6.144
    
    Parameters
    __________
    image : 3d array, float32
        0 axis is for different images that you want to CTF_calculate
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel Number: array 
        fresnel numbers for corresponding pictures
        can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    zero_compensation: int
        just a parameter, should be small, for ex. 0.01
    """
    
    nom = numpy.zeros((image.shape[1],image.shape[2]), dtype = 'complex128')
    denom = numpy.zeros((image.shape[1],image.shape[2]), dtype = 'complex128')
    
    for i in arange(image.shape[0]):
        nominator, denominator, f = ctf_mono((image.shape[1],image.shape[2]), 
                                             fresnelNumber[i], fresnelNumber[i], 
                                             beta_delta, zero_compensation)
        nom += multiply(fft2(image[i,:,:]-1), nominator) 
        denom += denominator
        
    ctf_out = multiply(divide(nom, denom), f)  
    ctf_out[abs(ctf_out)<zero_compensation] = zero_compensation 

    reco = ifft2(ctf_out).real
   
    return reco
    


def circle(x,y,r):
    """
    This function creates a mask - 2D array with zeros outside a circular region and ones inside the circle.
    x is the xaxis of the coordinate system, y is the y axis. The third argument is the radius of the circle. 
    
    Parameters
    __________
    x : 1D array
        x axis
    y : 1D array
        y axis
    r: int
        radius of the mask circle    
    """
    
    X,Y = meshgrid(x,y)
    c = sqrt(square(X)+square(Y))
    c[c<=r]=1
    c[c>r]=0
    return c 



def ctf_mono(imageshape, fresnelx, fresnely, beta_delta, zero_compensation):
    """
    This is the main function which calculates nominator/denominator. Please refer to the book of Tim Salditt - Biomedical imaging (2017)
    """
    # set fourier coordinate system
    Ny = imageshape[0]
    Nx = imageshape[1] 
    dqx = 1/Nx
    dqy = 1/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
      
    # formulate the contrast transfer function
    argument = ifftshift(pi/fresnelx*square(Qx)+pi/fresnely*square(Qy))         # Салдит в своей книге по другому задает Qx, в 2пи раз больше
    ctf_mono_nominator = sin(argument)+ beta_delta * cos(argument)
    ctf_mono_denominator =  2 * square(ctf_mono_nominator)  
    
    # regularization    
    # cutoff matrix using the function from P. Cloetens
    cutoff_sigma = 0.01
    cutoff = sqrt(mean(fresnelx)*0.5)
    r = filt2(Ny, cutoff*Ny, cutoff_sigma*Ny, Nx, cutoff*Nx)  
    regularization = multiply(zero_compensation,(1-r))              
    ctf_mono_denominator = ctf_mono_denominator + regularization  
    
    # aliasing free region f
    N_correct = min(floor(fresnelx / dqx**2), floor(fresnely / dqy**2)) 
    f  =  ifftshift(circle(xx,yy,floor(N_correct/2)))
     
    return ctf_mono_nominator, ctf_mono_denominator, f
    


def filt2(n, cutn, sigman, m=None, cutm=None):
        
# =============================================================================
# # regularisation function
# =============================================================================

# =============================================================================
# % Copyright (C) 2007 P. Cloetens
# %
# % This program is free software; you can redistribute it and/or modify
# % it under the terms of the GNU General Public License as published by
# % the Free Software Foundation; either version 2 of the License, or
# % (at your option) any later version.
# %
# % This program is distributed in the hope that it will be useful,
# % but WITHOUT ANY WARRANTY; without even the implied warranty of
# % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# % GNU General Public License for more details.
# %
# % You should have received a copy of the GNU General Public License
# % along with this program; if not, write to the Free Software
# % Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
# %
# % filt2
# %		r = filt2(n,cutn,sigman,varargin)
# %		create low-pass 2D filter in Fourier domain
# %		erfc type
# %
# %		arguments:
# %		argument 1: number of pixels along dimension 1
# %		argument 2: cut_off along dimension 1
# %		argument 3: width (sigma) of transition along dimension 1
# %		argument 4: number of pixels along dimension 2 ( default : same as argument 1)
# %		argument 5: cut_off along dimension 2 ( default : isotropic )
# %
# %		See also: filt2_gaussian
# 
# % Author: P. Cloetens <cloetens@esrf.fr>
# %
# % 2007-03-26 P. Cloetens <cloetens@esrf.fr>
# % * Initial revision
# %
# % adapted to Matlab: Martin Krenkel, February 2013
# =============================================================================
    
    if not m:
        m = n
    if not cutm:    
        cutm = cutn*m/n
        
        
    x,y = meshgrid(arange(floor(m/2)+1),arange(floor(n/2)+1))
    r = sqrt(square(x)+square(y*(cutm/cutn)))
 
    # im Prinzip nur hier:
    r = erfc((r-cutn)/sigman)
    r=r/r[0,0]
    
    r1 = fliplr(r[:,1:int(ceil(m/2))])
    r2 = flipud(r[1:int(ceil(n/2)),:]) 
    r3 = rot90(r[1:int(ceil(n/2)),1:int(ceil(m/2))],2)
    
    out0 = concatenate((r,r1), axis=1)
    out1 = concatenate((r2,r3), axis=1)
    out  = concatenate((out0,out1), axis=0)
        
    return out







