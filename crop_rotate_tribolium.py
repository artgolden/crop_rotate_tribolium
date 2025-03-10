import datetime
import json
import multiprocessing
import subprocess
import sys
import warnings
import tifffile as tiff
import os
import shutil
import re
import argparse
import logging
from dataclasses import dataclass, field
from typing import List
import io
from tqdm import tqdm
import numpy as np
from skimage import measure
from skimage import filters
from skimage import morphology
from skimage import transform
from skimage import draw
import cv2
from scipy import ndimage as cpu_ndimage
from typing import Optional

RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO = 1.15


def logging_broadcast(message):
    logging.info(message)
    print(message)

def load_image(file_path):
    """
    Args:
        file_path (str): The path to the TIFF file.

    Returns:
        numpy.ndarray: An 8-bit or 16-bit numpy array representing the 3D volume.
                       Returns None if the file cannot be loaded or if the data type is unsupported.
    """
    try:
        image = tiff.imread(file_path)

        if image.dtype == np.uint8 or image.dtype == np.uint16:
            return image
        else:
            print(
                f"Unsupported data type: {image.dtype}.  Only uint8 and uint16 are supported."
            )
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading TIFF file: {e}")
        return None

def load_and_merge_illuminations(ill_file_paths: list[str]):
    images = [load_image(f) for f in ill_file_paths]
    assert len(images) < 3 # There should be no more than 2 illuminations possible for each volume
    if not isinstance(images[0], np.ndarray):
        print(f"Error loading first volume from files: {ill_file_paths}")
        return None
    if len(images) == 1:
        return images[0]
    else:
        return np.mean(np.stack(images, axis=0), axis=0).astype(images[0].dtype)

def threshold_2d_image_xy(image: np.ndarray):
    if image.shape > 2:
        raise ValueError("Input volume must be 2D.")
    img_median = filters.median(image, morphology.disk(5)) # TODO: need to replace with something based on pixel size in um

    th = filters.threshold_triangle(img_median)
    mask = img_median >= th

    def get_matrix_with_circle(radius, shape=None, center=None):
        """
        Creates a NumPy array with a filled circle of 1s, using skimage.draw.disk.

        Args:
            radius (int): The radius of the circle.
            shape (tuple of ints): The shape of the 2D array (rows, cols). Default square of radius*2+1.
            center (tuple of ints, optional): The center of the circle (row, col).
                                                If None, defaults to the center of the array.

        Returns:
            numpy.ndarray: A NumPy array representing the circle.
        """
        if shape is None:
            shape = (radius * 2 + 1, radius * 2 + 1)
        img = np.zeros(shape, dtype=np.uint8)

        if center is None:
            center = (shape[0] // 2, shape[1] // 2)  # Integer division for center

        rr, cc = draw.disk(center, radius, shape=shape) # Get circle indices within bounds
        img[rr, cc] = 1

        return img


    # Clean up the mask
    cleaning_circle_radius = round(mask.shape[1] * 0.014)
    structuring_element = get_matrix_with_circle(cleaning_circle_radius)
    mask = cpu_ndimage.binary_opening(mask, structure=structuring_element, iterations=5).astype(mask.dtype)
    return mask

def crop_rotated_rectangle(image: np.ndarray, center: tuple, size: tuple, angle: float) -> np.ndarray:
    """
    Crop out a rotated rectangle from a 2D image.

    The function rotates the image around the given center using the specified angle
    and then crops the rectangle with the specified size. The rectangle is defined by
    its center, width, height, and the rotation angle (in degrees, counter-clockwise).

    Parameters:
        image (np.ndarray): 2D input image.
        center (tuple): (x, y) coordinates for the center of the rectangle.
        size (tuple): (width, height) of the rectangle.
        angle (float): Rotation angle in degrees (counter-clockwise).

    Returns:
        np.ndarray: Cropped rotated rectangle image.
    """
    # Ensure center and size are in float format for precision
    center = (float(center[0]), float(center[1]))
    width, height = size

    # Obtain the rotation matrix for the given center and angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the entire image using the rotation matrix
    # The output image will have the same dimensions as the input image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]),
                                   flags=cv2.INTER_CUBIC)
    
    # Compute the top-left coordinate of the cropping rectangle
    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)
    w = int(width)
    h = int(height)

    # Crop the rotated image
    cropped = rotated_image[y:y+h, x:x+w]
    return cropped

def crop_around_embryo(image, mask, embryo_head_direction: str, target_crop_shape=None) -> Optional[np.ndarray]:
    # Convert boolean mask to uint8 - necessary for OpenCV
    binary_image = mask.astype(np.uint8) * 255

    # 2. Find connected components (objects)
    labels = measure.label(binary_image)
    regions = measure.regionprops(labels)

    if not regions:
        print("Error: No objects found in the image.")
        return None

    largest_region = max(regions, key=lambda region: region.area)

    # 3. Extract largest object and find its edges using Canny
    largest_object_mask = (labels == largest_region.label).astype(np.uint8) * 255  # Create a mask of the largest object

    edges = cv2.Canny(largest_object_mask, 100, 200) # Apply Canny edge detection

    y, x = np.where(edges == 255)  # (row, col) for white pixels
    points = np.column_stack((x, y))  # fitEllipse expects (x, y) order

    # 2. Check that we have enough points to fit an ellipse
    if len(points) < 5:
        print("Error: Not enough edge points to fit an ellipse for embryo segmentation.")
        return None

    # 3. Fit the ellipse
    rotated_rect = cv2.fitEllipse(points)

    full_depth = image.shape[0]
    # 4. Extract ellipse properties from RotatedRect
    center, (width, height), angle_deg = rotated_rect
    expand_r = RATIO_FOR_EXPANDING_THE_CROPPED_REGION_AROUND_THE_EMBRYO
    size = (int(width)*expand_r, int(height)*expand_r)  
    logging.info(f"Cropping embryo with center: {center}, size: {size}, angle: {angle_deg}")
    if target_crop_shape is not None:
        size = target_crop_shape
        logging.info(f"Target crop shape was provided: {target_crop_shape}")
    return crop_rotated_rectangle(image, center, size, angle_deg)
    
def get_git_commit_hash(script_path):
    """
    Retrieves the current Git commit hash for the given script.

    Args:
        script_path:  The path to the Python script.  This is used to
                      determine the repository's root directory.

    Returns:
        The Git commit hash as a string, or None if not in a Git repository
        or if there's an error.
    """
    try:
        # Use git rev-parse --short HEAD to get the short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path),  # Important: Run command in script's dir!
            check=True  # Raise exception on non-zero return code
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Warning: Not a git repository or git command failed.")
        return None
    except FileNotFoundError:
        print("Warning: Git command not found.  Make sure Git is installed and in your PATH.")
        return None
    except Exception as e:
        print(f"Warning: Could not get git commit hash: {e}")
        return None

def copy_script_with_commit_hash(output_dir):
    """
    Copies the script to the output directory, adding the commit hash to the filename.

    Args:
        script_path: The path to the Python script to copy.
        output_dir: The directory to copy the script to.
        commit_hash: Optional. The commit hash to include in the filename.
                     If None, the script's original name is used.
    """
    try:
        script_path = os.path.abspath(__file__)
    except Exception as e:
        print(f"Warning: could not get this script path: {e}")
        return
    if script_path is None:
        print("Warning: could not get this script path.")
        return
    
    script_name = os.path.basename(script_path)
    name, ext = os.path.splitext(script_name)

    commit_hash = get_git_commit_hash(script_path)

    if commit_hash:
        new_script_name = f"{name}_{commit_hash}{ext}"
    else:
        new_script_name = script_name

    output_path = os.path.join(output_dir, new_script_name)

    try:
        shutil.copy2(script_path, output_path)  # copy2 preserves metadata
        print(f"Script copied to: {output_path}")
    except Exception as e:
        print(f"Warning: could not copy source code of the script: {e}")



@dataclass()
class FileMetadata:
    filepaths: List[str] = field(default_factory=list) 
    timeseries_key: str = ""
    timepoint: int = 0
    specimen: int = 0
    illuminations: List[int] = field(default_factory=list) 
    embryo_head_direction: str = ""

def parse_filename(filepath: str):
    """
    Parse a TIF file name of the form:
    timelapseID-20241008-143038_SPC-0001_TP-0870_ILL-0_CAM-1_CH-01_PL-(ZS)-outOf-0073.tif

    Returns:
        timeseries_key: a string identifying the time series (all parts except the TP and ILL parts)
        timepoint: integer value parsed after _TP-
        illumination: integer value parsed after _ILL-
        specimen: integer value parsed after _SPC-
    """
    base = os.path.basename(filepath)
    tp_match = re.search(r'_TP-(\d+)', base)
    ill_match = re.search(r'_ILL-(\d+)', base)
    sp_match = re.search(r'_SPC-(\d+)', base)
    if not (tp_match and ill_match and sp_match):
        raise ValueError(f"Filename {base} does not match expected pattern.")
    
    timepoint = int(tp_match.group(1))
    illumination = int(ill_match.group(1))
    specimen = int(sp_match.group(1))
    
    # Remove TP and ILL parts to form the timeseries key
    timeseries_key = re.sub(r'_TP-\d+', '', base)
    timeseries_key = re.sub(r'_ILL-\d+', '', timeseries_key)
    # Remove file extension
    timeseries_key = os.path.splitext(timeseries_key)[0]
    
    return timeseries_key, timepoint, illumination, specimen

def group_files(file_list, specimen_filter, embryo_head_direction):
    """
    Processes a list of file paths and returns a nested dictionary of FileMetadata objects.
    Only files with a specimen id matching specimen_filter are processed.
    The embryo_head_direction from configuration is assigned to each FileMetadata.

    Returns:
        A dictionary with keys as timeseries_key and values as dictionaries of timepoint: FileMetadata.
    """
    series_dict = {}
    for f in file_list:
        try:
            key, tp, ill, sp = parse_filename(f)
        except ValueError as e:
            logging.warning(f"Failed to parse filename {f}: {e}")
            continue
        # Filter out files that do not match the desired specimen id.
        if sp != specimen_filter:
            continue
        metadata = FileMetadata(
            filepaths=[f],
            timeseries_key=key,
            timepoint=tp,
            specimen=sp,
            illuminations=[ill],
            embryo_head_direction=embryo_head_direction
        )
        if key not in series_dict:
            series_dict[key] = {}
        if tp not in series_dict[key]:
            series_dict[key][tp] = metadata
        else:
            series_dict[key][tp].filepaths.append(f)
            series_dict[key][tp].illuminations.append(ill)
    return series_dict

def process_timepoint(file_metadata: FileMetadata, output_dir: str, target_crop_shape=None, do_save_thresholding_mask=False):
    ill_file_paths = file_metadata.filepaths
    timepoint = file_metadata.timepoint
    embryo_head_direction = file_metadata.embryo_head_direction
    logging.info(f"Processing timepoint {timepoint} for specimen {file_metadata.specimen} with files: {ill_file_paths}")
    print(f"Processing timepoint {timepoint}")
    
    # Merge illuminations for the timepoint
    merged_image = load_and_merge_illuminations(ill_file_paths)
    if merged_image is None:
        logging.error(f"Error loading merged volume for files: {ill_file_paths}")
        return None

    # Threshold to get mask
    mask = threshold_2d_image_xy(merged_image)
    if do_save_thresholding_mask:
        mask_dir = os.path.join(output_dir, "thresholding_mask")
        os.makedirs(mask_dir, exist_ok=True)
        tiff.imwrite(os.path.join(mask_dir, f"thresholding_mask_tp_{timepoint}_sp_{file_metadata.specimen}.tif"), mask)
    # Crop around embryo. For the first timepoint, we call without target_crop_shape.
    # For subsequent timepoints, crop_around_embryo should use the provided target shape.
    if target_crop_shape is None:
        cropped_img = crop_around_embryo(merged_image, mask, embryo_head_direction)
    else:
        cropped_img = crop_around_embryo(merged_image, mask, embryo_head_direction, target_crop_shape)
    
    if cropped_img is None:
        logging.error(f"Error segmenting and cropping around embryo for files: {ill_file_paths}")
        return None
    cropped_shape = cropped_img.shape   
    logging.info(f"TP: {timepoint} Cropped volume shape: {cropped_shape}")
    
    filename = ill_file_paths[0]
    filename = re.sub(r"(PL-.*)\.(tif|tiff)", r"\1_z_proj_cropped_rotated.\2", filename)
    filename = re.sub(r"_ILL-.*_CAM", r"_ILL-merged_CAM", filename)
    tiff.imwrite(os.path.join(output_dir, filename), cropped_img)

    return cropped_shape

def process_time_series(timeseries_key: str, timepoints_dict: dict, base_out_dir: str):
    """
    Process a single time series in parallel for all timepoints after the first one.
    """
    series_out_dir = os.path.join(base_out_dir, timeseries_key)
    os.makedirs(series_out_dir, exist_ok=True)
    logging.info(f"Processing time series: {timeseries_key}")
    print(f"Processing time series: {timeseries_key}")

    # Process timepoints in ascending order.
    sorted_timepoints = sorted(timepoints_dict.keys())

    # Process the first timepoint sequentially to determine the target crop shape.
    first_tp = sorted_timepoints[0]
    logging.info(f"Processing first timepoint: {first_tp}")
    print(f"Processing first timepoint: {first_tp}")
    target_crop_shape = process_timepoint(timepoints_dict[first_tp], series_out_dir, target_crop_shape=None)
    if target_crop_shape is None:
        logging.error(f"Error processing first timepoint {first_tp}. Aborting processing of time series {timeseries_key}.")
        return

    # Prepare the list of remaining timepoints.
    remaining_timepoints = sorted_timepoints[1:]
    
    # Define a worker function for parallel processing.
    def worker(tp):
        return process_timepoint(timepoints_dict[tp], series_out_dir, target_crop_shape=target_crop_shape)

    # Process the remaining timepoints in parallel.
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(worker, remaining_timepoints),
            total=len(remaining_timepoints),
            desc=f"Series {timeseries_key}",
            unit="timepoint"
        ))

    # Check for errors in processing the parallel tasks.
    if any(result is None for result in results):
        logging.error(f"Error processing one or more timepoints in series {timeseries_key}.")
    else:
        logging.info(f"Successfully processed all timepoints in series {timeseries_key}.")

def main():
    parser = argparse.ArgumentParser(
        description="Process embryo TIF images using a configuration JSON file."
    )
    parser.add_argument("config_file", type=str, help="Path to JSON configuration file")
    parser.add_argument("output_folder", type=str, help="Output folder")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc.)")
    parser.add_argument("--skip_patterns", type=str, nargs='*', default=[], 
                        help="List of patterns; time series whose keys contain any of these will be skipped.")
    parser.add_argument("--reuse_peeling", action="store_true", help="Reuse peeling masks from the first timepoint")
    args = parser.parse_args()
    
    # Setup output folder and logging.
    os.makedirs(args.output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_folder, f"process_{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Starting processing from configuration file")
    print("Starting processing...")
    
    # Read configuration JSON file.
    try:
        with open(args.config_file, "r") as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Error reading config file {args.config_file}: {e}")
        print(f"Error reading config file {args.config_file}: {e}")
        sys.exit(1)
    
    # Validate configuration structure.
    if "timeseries" not in config or not isinstance(config["timeseries"], list):
        logging.error("Configuration file must contain a 'timeseries' key with a list of time series definitions.")
        print("Invalid configuration file. Expected structure:\n" +
              json.dumps({
                  "timeseries": [
                      {
                          "z_projections_folder": "/path/to/z_projections",
                          "specimen_id": 3,
                          "embryo_head_direction": "left"
                      }
                  ]
              }, indent=4))
        sys.exit(1)
    
    valid_config_entries = []
    for entry in config["timeseries"]:
        if not isinstance(entry, dict):
            logging.warning("Skipping non-dictionary entry in timeseries list.")
            continue
        required_keys = ["z_projections_folder", "specimen_id", "embryo_head_direction"]
        if any(k not in entry for k in required_keys):
            logging.warning(f"Skipping entry missing required keys. Entry: {entry}")
            continue
        folder = entry["z_projections_folder"]
        specimen_id = entry["specimen_id"]
        direction = entry["embryo_head_direction"]
        if not isinstance(specimen_id, int) or not (0 <= specimen_id <= 1000):
            logging.warning(f"Skipping entry with invalid specimen_id {specimen_id}. Must be an integer between 0 and 1000.")
            continue
        if direction not in ["left", "right"]:
            logging.warning(f"Skipping entry with invalid embryo_head_direction {direction}. Must be 'left' or 'right'.")
            continue
        if not os.path.isdir(folder):
            logging.warning(f"Skipping entry because folder does not exist: {folder}")
            continue
        valid_config_entries.append(entry)
    
    if not valid_config_entries:
        logging.error("No valid configuration entries found. Exiting.")
        print("No valid configuration entries found in configuration file. Please check the configuration file. Expected structure:\n" +
              json.dumps({
                  "timeseries": [
                      {
                          "z_projections_folder": "/path/to/z_projections",
                          "specimen_id": 3,
                          "embryo_head_direction": "left"
                      }
                  ]
              }, indent=4))
        sys.exit(1)
    
    logging.info(f"Found {len(valid_config_entries)} valid configuration entries.")
    
    # Optionally, copy the script with commit hash (if implemented).
    copy_script_with_commit_hash(args.output_folder)
    
    # Process each valid configuration entry.
    for entry in valid_config_entries:
        folder = entry["z_projections_folder"]
        specimen_id = entry["specimen_id"]
        direction = entry["embryo_head_direction"]
        logging.info(f"Processing folder {folder} for specimen {specimen_id} with embryo head direction '{direction}'.")
        print(f"Processing folder {folder} for specimen {specimen_id} with embryo head direction '{direction}'.")
        
        # Find all TIF files in the specified folder.
        tif_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif")]
        logging.info(f"Found {len(tif_files)} TIF files in {folder}.")
        if not tif_files:
            logging.error(f"No TIF files found in folder {folder}. Skipping.")
            print(f"No TIF files found in folder {folder}. Skipping.")
            continue
        
        # Group files into time series and filter by specimen id.
        timeseries_dict = group_files(tif_files, specimen_filter=specimen_id, embryo_head_direction=direction)
        logging.info(f"Found {len(timeseries_dict)} time series in folder {folder} after filtering for specimen {specimen_id}.")
        
        # Process each time series.
        for series_key, tp_dict in timeseries_dict.items():
            if any(pattern in series_key for pattern in args.skip_patterns):
                logging.info(f"Skipping time series '{series_key}' due to matching skip pattern {args.skip_patterns}")
                continue
            process_time_series(series_key, tp_dict, args.output_folder)
    
    logging.info("Processing complete")
    print("Processing complete.")

if __name__ == "__main__":
    main()