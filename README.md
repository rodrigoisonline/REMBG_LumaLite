# REMBG_LumaLite

This is something new and hot! 

This script processes images to remove backgrounds and clean up white pixels using a pre-trained deep learning model and several and exclusive image processing techniques. 

Background Cleaner:

color_mask_pixel_removal1: Removes pixels of a specified color from an image by replacing them with transparent pixels.
blur_process_images: Applies a blur effect to the alpha channel of an image, useful for softening edges.
clean_white_pixels: Changes non-white pixels to black.
c_clean_white_pixels: Makes non-black pixels fully transparent.
Model Prediction:

bkpredict_u2net and predict_u2net: These functions use a pre-trained U2NET model to generate a mask for an input image. The mask is then saved and processed.
Image Processing Workflow:

img_process: This function runs the full pipeline: it processes the image to remove the background, cleans up white and black pixels, and applies a blur effect. It then computes a mask by comparing the processed image with the original.
Execution:

At the bottom, the img_process function is called with specific input and output paths, demonstrating how to use the script.
This script uses libraries such as OpenCV, PIL, and PyTorch to handle image processing and deep learning tasks. Make sure to adjust file paths and model loading based on your environment.

