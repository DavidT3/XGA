#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by McKenna Leichty 20/02/2023, 14:04. Copyright (c) The Contributors
import xga
#xga.NUM_CORES = 30

#from astropy.units import Quantity
#from astropy.visualization import LinearStretch
import numpy as np
import pandas as pd

from xga.sources import GalaxyCluster, NullSource
#from xga.samples import ClusterSample
#from xga.sas import evselect_image, eexpmap, emosaic
#from xga.utils import xmm_sky

#from xga.sources import PointSource
#from xga.sas import evselect_spectrum
#from xga.xspec import single_temp_apec, power_law

#need this for using OnDemand to plot images
#%matplotlib inline
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
#from matplotlib.colors import LogNorm
#from matplotlib.ticker import LogLocator, LogFormatter
#from matplotlib.colors import Normalize

from astropy.visualization import ImageNormalize, LogStretch
#from astropy.visualization.mpl_normalize import ImageNormalize
#from photutils.isophote import EllipseGeometry
#from photutils.aperture import EllipticalAperture
#import matplotlib.patches as patches
#from scipy.optimize import minimize

#cutting contours
#from xga.imagetools.misc import pix_rad_to_physical, physical_rad_to_pix
#from astropy.cosmology import Cosmology
#from astropy.units import Quantity
from astropy.cosmology import LambdaCDM
DEFAULT_COSMO = LambdaCDM(70, 0.3, 0.7)
#from xga.imagetools.profile import annular_mask

def contour_lvl(my_ratemap_data, flux_per, sigma, mask):
    """
    returns the flux level to make contours at
    
    flux_per : percent flux threshold should be at
    sigma : level of smoothness by gaussian
    
    """
    
    #smooth using a gaussian filter
    smoothed_array = gaussian_filter(my_ratemap_data*mask, sigma=sigma)

    #calculate the flux threshold based on the desired percentage
    total_flux = np.sum(smoothed_array)
    flux_threshold = flux_per * total_flux

    #sort the flattened array in descending order and accumulate the flux
    sorted_flux = np.sort(smoothed_array, axis=None)[::-1]
    cumulative_flux = np.cumsum(sorted_flux)

    #determine the contour level that corresponds to flux threshold
    contour_index = np.argmax(cumulative_flux >= flux_threshold) #argmax to find max index where flux is just above threshold
    contour_level = sorted_flux[contour_index]
    
    return contour_level, smoothed_array

def view_contours(my_ratemap_data, demo_src, flux_per, sigma, mask, cmap, smoothed_plot = False):

    contour_level, smoothed_array = contour_lvl(my_ratemap_data, flux_per = flux_per, sigma = sigma, mask = mask)
    
    #apply a logarithmic stretch to the image data
    norm = ImageNormalize(demo_src.get_combined_ratemaps().data, stretch=LogStretch())
    
    #if user wants a smoothed final image
    if smoothed_plot == True:
        #plot the original X-ray image
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(smoothed_array, cmap=cmap, norm=norm, origin = 'lower')
        plt.colorbar(im)
        plt.title('Smoothed')
        
        #adding contours to image
        contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
        
        #write a function here to get custom legend for as many contours as you want
        custom_legend = plt.Line2D([], [], color='red', label=f'{flux_per}% max flux')
        plt.legend(handles=[custom_legend])
        
        plt.show()
    
    #if user wants a non-smoothed final image
    else:
        #plot the original X-ray image
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(my_ratemap_data*demo_src.get_mask('r500')[0], cmap='viridis', norm=norm, origin = 'lower')
        plt.title('Non-Smoothed')
        plt.colorbar(im)
        
        #adding contours to image
        contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
        
        #write a function here to get custom legend for as many contours as you want
        custom_legend = plt.Line2D([], [], color='red', label=f'{flux_per}% max flux')
        plt.legend(handles=[custom_legend])
        plt.show()
        
    return contours

view_contours(my_ratemap_data, demo_src, flux_per = 0.50, sigma = 3, mask = demo_src.get_mask('r500')[0], cmap = 'viridis', smoothed_plot = False)





