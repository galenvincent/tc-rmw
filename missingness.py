import GOES
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as nd
import matplotlib.pyplot as plt

def get_training_images(id, filepath, tc_data_to_merge, time_limits):
    """
    Get the set of nicely formatted and pre-processed images for a specific storm.

    Arguments:
        - id (str): Storm ID.
        - filepath (str): Filepath pointing to the directory where GOES xr.datasets 
            are stored.
        - tc_data_to_merge (pd.DataFrame): Dataframe containing synopic-time dates,
             wind speed, and ID for this storm (or all storms).
        - time_limits (pd.DataFrame): Dataframe containing the time bounds that 
            we want to return images within. Should have index as storm ID, and
            columns 'min' and 'max' giving pd.datetime objects in each column.
    
    Returns:

    """
    goes = GOES.Stamp.from_filename(filepath + id +'-half-hr-1000.nc')

    goes_times = pd.DataFrame({
        'ID': id,
        'date': pd.to_datetime(goes.data.time.values).round('min')
    })

    min_time = time_limits.loc[id]['min']
    max_time = time_limits.loc[id]['max']
    time_indices = np.squeeze(np.argwhere(((goes_times.date >= min_time) & (goes_times.date <= max_time)).to_numpy()))
    time_indices = np.sort(time_indices)

    goes_times_filtered = goes_times.iloc[time_indices]
    goes_times_filtered['time_idx'] = time_indices

    merged = goes_times_filtered.merge(data_to_merge, on = ['ID', 'date'], how = 'left')
    merged.wind = merged.wind.interpolate()

    merged[['category', 'category_num']] = merged.apply(storm_category, axis = 1, result_type='expand')

    images = []
    nan_fracs = []

    for time_idx in merged.time_idx:
        image, nan_frac = stamp_at_time(goes, time_idx, 400)
        filled_image = sequential_fill(image)
        reshaped_image = reshape(filled_image, size = (200, 200))

        images.append(reshaped_image)
        nan_fracs.append(nan_frac)
        
    images = np.array(images)
    merged['nan_frac'] = np.array(nan_fracs)
    

    return (images, merged)

def storm_category(row):
    if row['wind'] < 64: # Tropical Storm
        return ('TS', 0)
    if 64 <= row['wind'] < 96: # Cat 1 or 2 (hurricane: H)
        return ('H', 1)
    if 96 <= row['wind']: # Cat 3, 4, 5 (major hurricane: MH)
        return ('MH', 2)

def stamp_at_time(goes_dataset, time_idx, radius):
    """
    Given a full xarray dataset representing GOES observations for a given storm
    in Trey's format, return a square stamp with specified radius at the specified
    time index. 

    Arguments:
        - goes_dataset (xr.dataset): As returned from e.g. GOES.Stamp.from_filename()
        - time_idx (int): Time index at which to return a stamp.
        - radius (float): Units of km. Must be <= 1000km.

    Returns:
        List of: [0] a np.ndarray containing the temperature stamp.
                 [1] the fraction of pixels in the stamp that are missing.
    """
    temp_full = goes_dataset.data.temperature[time_idx].values
    lat_center = goes_dataset.data.LATCENTER.values[time_idx]
    lon_center = goes_dataset.data.LONCENTER.values[time_idx]

    bounding_box = GOES.coord_from_distance(lat_center, lon_center, radius)

    lats = goes_dataset.data.lat.values + goes_dataset.data.LATSHIFT.values[time_idx]
    lons = goes_dataset.data.lon.values + goes_dataset.data.LONSHIFT.values[time_idx]

    lat_idx = np.squeeze(np.argwhere((lats < bounding_box['latHi']) & (lats > bounding_box['latLo'])))
    lon_idx = np.squeeze(np.argwhere((lons < bounding_box['lonHi']) & (lons > bounding_box['lonLo'])))

    temp_stamp = temp_full[(min(lat_idx) - 1):(max(lat_idx) + 2), (min(lon_idx) - 1):(max(lon_idx) + 2)]

    nan_frac = np.isnan(temp_stamp).sum()/(temp_stamp.shape[0] * temp_stamp.shape[1])

    return [temp_stamp, nan_frac]

def fill(data):
    """
    Replace the value of invalid 'data' cells by the value of the nearest valid data cell

    Arguments:
        data:    numpy array of any dimension

    Output: 
        Return a filled array. 
    """    
    invalid = np.isnan(data)

    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)

    return data[tuple(ind)]

def sequential_fill(img):
    """
    Fill in missing pixels in a sattelite image.

    Step 1: Apply the cv2.inpaint method, which utilizes navier stokes equations
            to determine the best fill values. Experimentation shows that this 
            method leaves some pixels un-filled, but fills in the vast majority
            in a good looking way.

    Step 2: Utilize scipy.ndimage to fill the remaning missing pixels with the 
            value of its nearest neighbor pixel.

    Arguments:
        img (np.ndarray): A 2D numpy array representing sattelite imagery.
    """
    img_mask = np.where(np.isnan(img), 1, 0).astype('uint8') # Mask of missing pixels
    img_painted = cv2.inpaint(img, img_mask, 1, cv2.INPAINT_NS) # Step 1
    img_filled = fill(img_painted) # Step 2

    return img_filled

def reshape(img, size):
    return cv2.resize(img, size, interpolation = cv2.INTER_AREA)

