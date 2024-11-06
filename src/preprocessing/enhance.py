import cv2, math
import numpy as np
from skimage import color
from PIL import Image

def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(img)
    return cl
    
def anisotropic_diffusion(img_array, num_iterations=10, kappa=30, gamma=0.1, option=2):
    """
    Applies anisotropic diffusion to a grayscale image represented as a numpy array.
    
    Parameters:
    - img_array: Input grayscale image as a numpy array (values 0-255)
    - num_iterations: Number of iterations to perform diffusion
    - kappa: Controls the sensitivity to edges (gradient threshold)
    - gamma: Diffusion coefficient
    - option: 1 or 2, selects conduction function (edge-stopping function)
    
    Returns:
    - diffused_img: Image after applying anisotropic diffusion as a numpy array
    """
    img_array = img_array.astype('float32')  # Convert to float for calculations
    diffused_img = img_array.copy()
    
    for i in range(num_iterations):
        # Shifted images (up, down, left, right) for neighbors
        north = np.roll(diffused_img, -1, axis=0)
        south = np.roll(diffused_img, 1, axis=0)
        east = np.roll(diffused_img, -1, axis=1)
        west = np.roll(diffused_img, 1, axis=1)
        
        # Calculate gradients
        delta_north = north - diffused_img
        delta_south = south - diffused_img
        delta_east = east - diffused_img
        delta_west = west - diffused_img
        
        # Conduction functions (edge-stopping functions)
        if option == 1:
            c_north = np.exp(-(delta_north / kappa) ** 2)
            c_south = np.exp(-(delta_south / kappa) ** 2)
            c_east = np.exp(-(delta_east / kappa) ** 2)
            c_west = np.exp(-(delta_west / kappa) ** 2)
        elif option == 2:
            c_north = 1.0 / (1.0 + (delta_north / kappa) ** 2)
            c_south = 1.0 / (1.0 + (delta_south / kappa) ** 2)
            c_east = 1.0 / (1.0 + (delta_east / kappa) ** 2)
            c_west = 1.0 / (1.0 + (delta_west / kappa) ** 2)
        
        # Update the image with anisotropic diffusion formula
        diffused_img += gamma * (
            c_north * delta_north +
            c_south * delta_south +
            c_east * delta_east +
            c_west * delta_west
        )
    
    return np.clip(diffused_img, 0, 255).astype('uint8')