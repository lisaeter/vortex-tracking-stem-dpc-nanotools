import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import disk
from skimage.util import img_as_float

def main(original_images, registered_images):
    # Use reference image (0 degrees tilt)
    image = img_as_float(registered_images["002"])
    
    # Detect edges using Canny
    edges = canny(image, sigma=2, low_threshold=0.04, high_threshold=0.05)
    
    # Possible disk radii (pixels)
    hough_radii = np.arange(8, 11, 1)
    
    # Calculate Hough transform
    hough_res = hough_circle(edges, hough_radii)
    
    # Find the strongest circle candidates
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        total_num_peaks=68,
        min_xdistance=25,
        min_ydistance=25
    )
    
    print(f"Detected circles: {len(cx)}")

    # Sort circles by x-coordinate, from right to left
    indices = np.argsort(-cx)
    
    # Group circles into columns
    columns = []
    current_column = [indices[0]]
    
    tolerance = 15  # Maximum x-deviation within a column (pixels)
    
    for idx in indices[1:]:
    
        # Compare with mean x-position of current column
        mean_x = np.mean(cx[current_column])
    
        if abs(cx[idx] - mean_x) < tolerance:
            current_column.append(idx)
    
        else:
            columns.append(current_column)
            current_column = [idx]
    
    columns.append(current_column)
    
    # Sort each column from top to bottom
    sorted_indices = []
    
    for column in columns:
        column_sorted = sorted(column, key=lambda i: cy[i])
        sorted_indices.extend(column_sorted)
    
    # Reorder detected circles
    cx = cx[sorted_indices]
    cy = cy[sorted_indices]
    radii = radii[sorted_indices]
    accums = accums[sorted_indices]

    image_shape = registered_images["002"].shape

    disk_masks = []
    
    for x, y, r in zip(cx, cy, radii):
    
        mask = np.zeros(image_shape, dtype=bool)
    
        rr, cc = disk((y, x), r, shape=image_shape)
        mask[rr, cc] = True
    
        disk_masks.append(mask)
    
    # disk_masks_array = np.stack(disk_masks)
    
    print(f"Created {len(disk_masks)} disk masks")

    return disk_masks