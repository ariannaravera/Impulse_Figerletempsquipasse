from datetime import datetime, timedelta
from dateutil.parser import parse
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import sys

"""def date_to_seconds(date_str):
    try:
        date_obj = parse(date_str)
        return (date_obj - datetime.utcfromtimestamp(0)).total_seconds()
    except:
        return np.nan

def seconds_to_date(seconds):
    if np.isnan(seconds):
        return "NaN"
    return (datetime.utcfromtimestamp(0) + timedelta(seconds=seconds)).strftime("%Y-%m-%d")
"""

def create_mozaic(image_width, image_height, num_bins_x, num_bins_y, min_id, max_id, id_list):

    mozaic_width = image_width / num_bins_x
    mozaic_height = image_height / num_bins_y

    if mozaic_width % 1 == 0 and mozaic_height % 1 == 0:
        mozaic_width = int(mozaic_width)
        mozaic_height = int(mozaic_height)
    else:
        print("Please make sure that the ratios (image_width / num_bins_x) and (image_height / num_bins_y) are integers")
        sys.exit()

    x_bins = np.linspace(0, 1, num_bins_x + 1)
    y_bins = np.linspace(0, 1, num_bins_y + 1)

    mozaic = np.full((num_bins_x, num_bins_y), np.nan)

    for x, y, date_id in id_list:
        i = int(x * num_bins_x)
        j = int(y * num_bins_y)
        k = int(date_id)
        mozaic[i, j] = k

    x_indices, y_indices = np.meshgrid(np.arange(num_bins_x), np.arange(num_bins_y))
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    mozaic_values = np.transpose(mozaic).flatten()

    # Valid indices contain float values given as input
    valid_indices = ~np.isnan(mozaic_values)
    valid_x_indices = x_indices[valid_indices]
    valid_y_indices = y_indices[valid_indices]
    valid_idss = mozaic_values[valid_indices]

    # Interpolate values for all grid points with a cubic spline
    interpolated_id = griddata((valid_x_indices, valid_y_indices),
                                  valid_idss,
                                  (x_indices, y_indices),
                                  method='cubic', fill_value=np.nan )

    # The cubic method may leave some NaN values, so we use a simpler method to fill in these remaining NaNs
    if(np.isnan(interpolated_id).any()):

        # Valid indices contain float values given as input + spline interpolation
        valid_indices = ~np.isnan(interpolated_id)
        valid_x_indices = x_indices[valid_indices]
        valid_y_indices = y_indices[valid_indices]
        valid_ids = interpolated_id[valid_indices]

        # Interpolate the remaining NaNs values with the nearest method
        interpolated_id = griddata((valid_x_indices, valid_y_indices), 
                                      valid_ids,
                                        (x_indices, y_indices),
                                        method='nearest', fill_value=np.nan )

    # Clip the interpolated ids to the specified range
    interpolated_id = np.clip(interpolated_id, min_id, max_id)

    """fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(x_bins, y_bins, interpolated_id.reshape(num_bins_y, num_bins_x), shading='auto')
    plt.colorbar(c, label='Date (seconds)')
    plt.xlabel('X Bins')
    plt.ylabel('Y Bins')
    plt.title('Mozaic Interpolation')
    plt.savefig('mozaic.png')
    plt.close(fig)"""

    result = []
    for i, j in zip(x_indices, y_indices):
        value = interpolated_id[i + j * num_bins_x]
        for x_delta in range(mozaic_width):
            for y_delta in range(mozaic_height):
                x_pixel = i * mozaic_width + x_delta
                y_pixel = j * mozaic_height + y_delta
                result.append((x_pixel, y_pixel, value))

    sorted_result = sorted(result, key = lambda x: (x[1], x[0]))

    return sorted_result

"""# How to use it #
image_width = 20 # the number of pixels for the width of the output image
image_height = 20 # the number of pixels for the height of the output image
num_bins_x = 10 # the number of horizontal "boxes" in the output mozaic image
num_bins_y = 10  # the number of vertical "boxes" in the output mozaic image
min_date = "2021-01-01" # the smallest acceptable date
max_date = "2023-12-31" # the largest acceptable date
# the positions (x, y) must be values within [0, 1] x [0, 1] and they should NOT be collinear, e.g.
# BAD: date_list = [(0.1, 0.1, "2021-09-26"), (0.5, 0.5, "2022-05-02"), (0.9, 0.9, "2022-03-27")]
# GOOD: date_list = [(0.15, 0.1, "2021-09-26"), (0.3, 0.5, "2022-05-02"), (0.7, 0.9, "2022-03-27")]
#date_list = [(0.1, 0.1, "2021-09-26"), (0.1, 0.9, "2023-09-27"), (0.5, 0.5, "2022-05-02"),  (0.9, 0.1, "2021-03-27"), (0.9, 0.9, "2022-03-27")]
date_list = [(0.01, 0.01, "2021-09-26"),(0.01, 0.99, "2021-09-26"), (0.99,0.01, "2022-03-27"), (0.99, 0.99, "2022-03-27")]

mozaic = create_mozaic(image_width, image_height, num_bins_x, num_bins_y, min_date, max_date, date_list)
#print(mozaic)"""
