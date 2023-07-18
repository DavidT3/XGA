#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by McKenna Leichty 20/02/2023, 14:04. Copyright (c) The Contributors

#imported modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.visualization import ImageNormalize, LogStretch

from typing import Tuple, List, Union
from ..products import Image, RateMap, ExpMap

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

def view_contours(im_prod: Union[Image, RateMap, np.ndarray], demo_src, flux_per, sigma, mask, cmap, smoothed_plot = False, masked = False):

    if isinstance(im_prod, (Image, RateMap)):
        # For the XGA Image or RateMap products
        my_ratemap_data = im_prod.data
    elif isinstance(im_prod, np.ndarray):
        # For numpy arrays
        my_ratemap_data = im_prod  

    #my_ratemap_data = demo_src.get_combined_ratemaps().data
    contour_level, smoothed_array = contour_lvl(my_ratemap_data, flux_per = flux_per, sigma = sigma, mask = mask)
    
    #apply a logarithmic stretch to the image data
    norm = ImageNormalize(my_ratemap_data, stretch=LogStretch())
    
    #if user wants a smoothed final image
    if smoothed_plot == True:
        #if user wants the smoothed image to be masked
        if masked == True:
            #plot the original X-ray image
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(smoothed_array*mask, cmap=cmap, norm=norm, origin = 'lower')
            plt.colorbar(im)
            plt.title(f'Smoothed - {demo_src.name}')
            
            #adding contours to image
            contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
            
            #write a function here to get custom legend for as many contours as you want
            custom_legend = plt.Line2D([], [], color='red', label=f'{100*flux_per}% max flux')
            plt.legend(handles=[custom_legend])
            
            plt.show()
        
        else: #not masked
            #plot the original X-ray image
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(smoothed_array, cmap=cmap, norm=norm, origin = 'lower')
            plt.colorbar(im)
            plt.title(f'Smoothed - {demo_src.name}')
            
            #adding contours to image
            contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
            
            #write a function here to get custom legend for as many contours as you want
            custom_legend = plt.Line2D([], [], color='red', label=f'{100*flux_per}% max flux')
            plt.legend(handles=[custom_legend])
            
            plt.show()
        
    #if user wants a non-smoothed final image
    else:
        #if user wants a masked non-smoothed image
        if masked == True:
            #plot the original X-ray image
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(my_ratemap_data*mask, cmap=cmap, norm=norm, origin = 'lower')
            plt.colorbar(im)
            plt.title(f'Non-Smoothed - {demo_src.name}')
            
            #adding contours to image
            contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
            
            #write a function here to get custom legend for as many contours as you want
            custom_legend = plt.Line2D([], [], color='red', label=f'{100*flux_per}% max flux')
            plt.legend(handles=[custom_legend])
            plt.show()
        
        else: #not masked
            #plot the original X-ray image
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(my_ratemap_data, cmap=cmap, norm=norm, origin = 'lower')
            plt.colorbar(im)
            plt.title(f'Non-Smoothed - {demo_src.name}')
            
            #adding contours to image
            contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
            
            #write a function here to get custom legend for as many contours as you want
            custom_legend = plt.Line2D([], [], color='red', label=f'{100*flux_per}% max flux')
            plt.legend(handles=[custom_legend])
            
            plt.show()
        
    return contours





