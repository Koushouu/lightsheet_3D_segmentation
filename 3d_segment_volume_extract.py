import random
import numpy as np
from skimage.io import imsave, imread
from skimage.morphology import binary_erosion, binary_dilation
import pandas
# USER INPUTS FIELD ##################################################################
# Your path to npy file here
NPY_FILE = r'U:\Kou\Vicky_kidney\original_vol\tut\raw_img_bin2_seg.npy'
# Your coordinates file here
CSV_FILE = r'U:\Kou\Vicky_kidney\original_vol\tut\220914_ball_tub.csv'
# The file name for segmented volume here
SEGMENTED_VOLUME = r'U:\Kou\Vicky_kidney\original_vol\tut\balls_220914_ball_tub.tif'
######################################################################################
MASKS_FILE = NPY_FILE[:-8] + '_masks.tif'
OUTLINES_RANDID_FILE = NPY_FILE[:-8] + '_randcolor_outlines.tif'
OUTLINES_FILE = NPY_FILE[:-8] + '_outlines.tif'

UPDATED_OUTLINES_FILE = NPY_FILE[:-8] + '_randcolor_outlines_updated.tif'
# Read data
print("Reading data...")
masks = imread(MASKS_FILE)
outlines = imread(OUTLINES_RANDID_FILE)
outlines_raw = imread(OUTLINES_FILE)
coordinates = pandas.read_csv(CSV_FILE)
coordinates = coordinates.to_dict(orient='records')

# Map coordinates on masks and extract IDs
print("Extracting IDs...")
roi_id = []
for coordinate in coordinates:
    x_cor = coordinate['X']
    y_cor = coordinate['Y']
    z_cor = coordinate['Slice']
    mask_id = masks[z_cor, y_cor, x_cor]
    if mask_id == 0:
        pass
    else:
        roi_id.append(mask_id)
# Only extract the unique IDs
roi_id = np.unique(roi_id)
# Loop through IDs to fill volumes in the zero matrix
print("Extracting volume...")
extracted_vol = np.zeros_like(masks)
n = random.randint(0, 255)
for id in roi_id:
    # Loop through IDs to fill volumes in the zero matrix
    extracted_vol[masks == id] = 1
    # Loop through IDs to assign same colour on the outlines
    outlines[outlines_raw == id] = n
# Dilation/ Erosion
print("Performing erosion/ dilation...")
extracted_vol = binary_erosion(extracted_vol)
for dil in range(3):
    print(f"Performing dilation #{str(dil)}")
    extracted_vol = binary_dilation(extracted_vol)

for ero in range(3):
    print(f"Performing erosion #{str(ero)}")
    extracted_vol = binary_erosion(extracted_vol)

# Write data
imsave(UPDATED_OUTLINES_FILE, outlines)
imsave(SEGMENTED_VOLUME, extracted_vol)
