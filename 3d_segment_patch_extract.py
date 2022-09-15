import numpy as np
from skimage.io import imsave
import random

# USER INPUTS FIELD ##################################################################
# Path to your npy file
NPY_FILE = r'U:\Kou\Vicky_kidney\original_vol\tut\raw_img_bin2_seg.npy'
######################################################################################
OUTLINES_RANDID_FILE = NPY_FILE[:-8] + '_randcolor_outlines.tif'
OUTLINES_FILE = NPY_FILE[:-8] + '_outlines.tif'
MASKS_FILE = NPY_FILE[:-8] + '_masks.tif'

# Read npy
print("Reading the npy file...")
npy = np.load(NPY_FILE, allow_pickle=True).item()
# ---------------------------------------Extract masks ----------------------------------------
print("Reading/ writing the mask file...")
masks = npy['masks']
imsave(MASKS_FILE, masks)
# ------------------------------------ Extract outlines ---------------------------------------
print("Reading the outlines file...")
outlines = npy['outlines']
# Renumerate outline IDs to below 8-bit, for visualization
print("Re-assigning the outline IDs...")
unique_ids = np.unique(outlines)
new_outlines = np.zeros_like(outlines)
total = len(unique_ids)
for i in unique_ids[1:]:
    random_id = random.randint(0, 255)
    new_outlines[outlines == i] = random_id
    print(f"{str(round(100 * i / total, 2))} % completed")
print("Writing the outlines file...")
imsave(OUTLINES_RANDID_FILE, new_outlines)
imsave(OUTLINES_FILE, outlines)
