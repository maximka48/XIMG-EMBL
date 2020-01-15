"""
Created on Fri Oct  12 15:46:49 2018

@author: mpolikarpov
"""


from numpy import floor, meshgrid, multiply, divide, square, sqrt, pi, arange, log, pad, exp
from numpy.fft import fftshift, ifftshift, fft2, ifft2



def Paganin(image, fresnel, beta_delta, zero_compensation = 0.01):
    """
    This function makes the phase retrieval with Paganin method. 
        Refer to the book:  Tim Salditt - Biomedical imaging - 2017 for references
    
    Parameters
    __________
    image     : float32
        2D image
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel   : int
        Fresnel Number - can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    zero_compensation: int
        just a parameter, should be small, for ex. 0.01
    
    """
    # set fourier coordinate system
    Ny = image.shape[0]
    Nx = image.shape[1] 
    dqx = 2*pi/Nx
    dqy = 2*pi/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
    
    k2 = (4*pi*fresnel)*beta_delta
    
    # formulate the equation
    nominator = 1 / (ifftshift(square(Qx)+square(Qy)) + k2)        
    dvd = ifft2(multiply(fft2(image), nominator)).real    
    phi_r = (1/(2 * beta_delta)) * log(k2 * dvd)

    return phi_r    
    
    
    
    
def MBA(image, fresnel, beta_delta, zero_compensation = 0.01):
    """
    This function makes the phase retrieval with Modified Bronnikov Algorithm. 
        Refer to the book:  Tim Salditt - Biomedical imaging - 2017 for references
    
    Parameters
    __________
    image     : float32
        2D image
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel   : int
        Fresnel Number - can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    zero_compensation: int
        just a parameter, should be small, for ex. 0.01
    
    """
    
    # set fourier coordinate system
    Ny = image.shape[0]
    Nx = image.shape[1] 
    dqx = 2*pi/Nx
    dqy = 2*pi/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
    
    k2 = (4*pi*fresnel)*beta_delta
    
    # formulate Modified Bronnikov formula
    nominator = 1 / (ifftshift(square(Qx)+square(Qy)) + k2)
    dvd = 2*pi*fresnel * ifft2(multiply(fft2(image-1), nominator)).real
    return dvd    
    




def BAC (image, fresnelNumber, alpha, gamma, zero_compensation = 0.01):
    
    """
    This function makes the Bronnikov aided correction 
    Please refer to the book of Tim Salditt "Biomedical imaging", 2017, ISBN = 978-3-11-042668-7, ISBN (PDF)= 978-3-11-0426694
    equation 6.144
    
    Parameters
    __________
    image : float32
        2D image
    beta_delta: int
        absorption divided by refraction  (decrements). Normally small, for ex. 0.08 
    fresnel Number: int
        can be calculated as fresnelN = pixelsize**2/(wavelength*object_detector_distance)
    zero_compensation: int
        just a parameter, should be small, for ex. 0.01
    """
    
    gamma = gamma/fresnelNumber
    phi_r = MBA(image, fresnelNumber, alpha, zero_compensation)
    denominator = 1 - gamma *  laplacian(phi_r)
    I0 = (divide(image, denominator)) 
    
    return I0







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


    
def laplacian(image, Npad = None):
    """
    This is Laplacian in fourier space
    
    Parameters
    __________
    image: 2D arrat
    Npad: int
        number to pad the image
    """
    
    if not Npad:
        Npad = [int(image.shape[0]/2), int(image.shape[1]/2)]
        
    # padding with border values
    image = pad(image, ((Npad[0], Npad[0]),(Npad[1], Npad[1])), 'edge')   
    
    # coordinate system in fourier space
    Ny = image.shape[0]
    Nx = image.shape[1] 
    dqx = 2*pi/Nx
    dqy = 2*pi/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
    
    # fourier representation of laplacian
    filt = -fftshift(square(Qx)+square(Qy))    

    #filter itself

    out = ifft2(multiply(fft2(image),filt))

    #undo padding and return result
    result = out[Npad[0]:(out.shape[0]-Npad[0]), Npad[1]:(out.shape[1]-Npad[1])].real           
    
    return result




def anka_filter(image, sigma, stabilizer, Npad = None):
    """
    unsharp filter from ankaphase programm - called "image_restoration" there
    
    Parameters
    __________
    image: 2D arrat
    sigma: float
        width of the gaussian kernel
    stabilizer: float
        small number
    Npad: int, optional
        number to pad the image
    """
    if not Npad:
        Npad = [int(image.shape[0]/2), int(image.shape[1]/2)]
     
# =============================================================================
#     # padding with border values
#     image = pad(image, ((Npad[0], Npad[0]),(Npad[1], Npad[1])), 'edge')   
# =============================================================================
        
    #set coordinate grid
    Ny = image.shape[0]
    Nx = image.shape[1]
    dqx = 2*pi/Nx
    dqy = 2*pi/Ny
    
    xx = arange(1,Nx+1)-floor(Nx/2)-1 
    yy = arange(1,Ny+1)-floor(Ny/2)-1
    Qx, Qy = meshgrid(xx*dqx, yy*dqy)
    
    #filtering
    nomin = 1 + stabilizer
    denomin = stabilizer + exp((-pi)*(sigma**2)*ifftshift(square(Qx)+square(Qy)))
    out = ifft2(multiply(fft2(image), divide(nomin, denomin)))
    
    #undo padding and return result
    #result = out[Npad[0]:(out.shape[0]-Npad[0]), Npad[1]:(out.shape[1]-Npad[1])].real 
    result = out.real 
    return result


