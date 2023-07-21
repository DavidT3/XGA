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
    Smoothes the image using a Gaussian filter and calculates the flux level 
    to make contours at.
    
    :param my_ratemap_data: the 2D array to find the contour levels of
    :param flux_per: the percentage of maximum flux to find the contour levels at
    :param sigma: used to change the level of smoothness via a gaussian filter
    :param mask: the mask to use if wanted on plot, and mask to smooth with gaussian filter (so as not to include pixels outside of cluster)
    
    :return: levels at which to plot the contours and a smoothed array to plot a smoothed final image
    :rtype: numpy.float64, np.array
    """
    
    #smooth using a gaussian filter after applying mask
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

def view_contours(im_prod: Union[Image, RateMap, np.ndarray], flux_per, sigma, mask, cmap, smoothed_plot = False, masked = False):
    """
    Takes an XGA product and produces contours at a specified max flux level.
    A mask or smoothing can be applied to the final image.

    :param Image/RateMap/ndarray im_prod: the 2D array or RateMap to find the contours of
    :param flux_per: the percentage of maximum flux to find the contour levels at
    :param sigma: used to change the level of smoothness via a gaussian filter
    :param mask: the mask to use if wanted on plot, and mask to smooth with gaussian filter (so as not to include pixels outside of cluster)
    :param cmap: colormap of final image
    :param smoothed_plot = False: if true, final image will be shown smoothed
    :param masked = False: if true, final image will show mask inputted by user
    
    :return: the contours produced at specified flux level
    :rtype: matplotlib.contour.QuadContourSet
    """

    if isinstance(im_prod, (Image, RateMap)):
        # For the XGA Image or RateMap products
        my_ratemap_data = im_prod.data
        name = im_prod.src_name
    elif isinstance(im_prod, np.ndarray):
        # For numpy arrays
        my_ratemap_data = im_prod
        name = ""

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
            plt.title(f'Smoothed - {name}')
            
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
            plt.title(f'Smoothed - {name}')
            
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
            plt.title(f'Non-Smoothed - {name}')
            
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
            plt.title(f'Non-Smoothed - {name}')
            
            #adding contours to image
            contours = plt.contour(smoothed_array, levels=[contour_level], colors='red')
            
            #write a function here to get custom legend for as many contours as you want
            custom_legend = plt.Line2D([], [], color='red', label=f'{100*flux_per}% max flux')
            plt.legend(handles=[custom_legend])
            
            plt.show()
    
    return contours

def contour_coords(contours):
    """
    Extracts the x and y points of the contours.
    
    :param contours: the contours produced at specified flux level
    :return: the x and y coordinates of contours generated
    :rtype: np.ndarray
    """
    #extract the contour coordinates
    x_points = []
    y_points = []
    for path in contours.collections[0].get_paths():
        vertices = path.vertices
        x = vertices[:, 0]
        y = vertices[:, 1]
        x_points.extend(x)
        y_points.extend(y)
    
    return x_points, y_points



