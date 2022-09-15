# lightsheet_3D_segmentation:
# Kidney segmentation in 3D

## Segmentation procedure

1. Cellpose processing. Execute the command line below in **anaconda prompt**. See [here](https://www.notion.so/Kidney-segmentation-in-3D-a0dd5c532a394b2fb7aec3f703d79def) for how I constructed this line of command, as you might need to change this line according to your samples.
    
    ```bash
    python -m cellpose --dir U:\Kou\Vicky_kidney\original_vol\bin2_3 --pretrained_model cyto2 --chan 0 --diameter 75.0 --flow_threshold 0.4 --cellprob_threshold 0.0 --verbose --use_gpu --stitch_threshold 0.1
    ```
    
2. Run the following script to extract mask and outline (**3D_segment_patch_extract.py**). **Remember to change the user input field** as you like
    
    ```python
    import numpy as np
    from skimage.io import imsave
    import random
    
    # USER INPUTS FIELD ################################################################
    # Path to your npy file
    NPY_FILE = r'U:\Kou\Vicky_kidney\original_vol\tut\raw_img_bin2_seg.npy'
    ####################################################################################
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
    ```
    
3. Open the `*_randcolor_outlines.tif` in Image J
4. (Optional) Dilate (Thicken) the outline. 
    1. Go to Image > Type > 8-bit
    2. Go to Process > Morphology > Grey Morphology
    3. Set Radius of the structure element to 1.0 (Or to as much as you think its enough) ; set Operator to dilate > then OK
    4. Once finished, go to Image > Type > 16-bit
5. Overlap the outlines with the raw image. Open the raw image in Image J
    1. Go to Image > Color > Merge channels
    2. Set the cellpose_outlines.tif as red, and the raw image as gray. Tick **Create composite** and **Keep source images**
    3. OK
6. Then you will see this
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled.png)
    
7. Now, go to Image > Color > Arrange Channels
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%201.png)
    
8. Right click on Old image [1] (as shown above), select **3-3-2 RGB.** And you will see this
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%202.png)
    
9. Select region of interest.
    1. Go to multipoints (See below)
        
        ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%203.png)
        
    2. Click on all regions of interest, across slices. For ROI with the same outline colors, you only have to click once. If you accidentally click twice, it wont be a problem. If you see a patch that you want but was not outlined, leave it there for now. 
    3. Hint: to go up/ down slices, hold [alt] key while scrolling. 
    4. To delet points, hold [ctrl] while clicking on the point
10. Once all the region of interests are clicked, go to Analyze > Set Measurements
11. Tick **Centroid**, **Stack position**, and untick everything else; Set Decimal places to 0
12. Select your image window, press [m], you will get a table like below
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%204.png)
    
13. Save as *.csv. On the Results window (See the red arrow above), Go to File > Save as…
14. Save again into a different data format. Go to File (Different “File”! See the red arrow below)> Save as> Selection… save coordinates as *.roi
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%205.png)
    
15. Run the following script (**3D_segment_extract_volume.py**), to extract the volume of interest
    
    ```python
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
    ```
    
16. Now you can open your segmented volume in Amira or any other softwares

## Additional annotation by ITK-SNAP

Cellpose might not be able to do everything perfectly and you might see a few ROIs unidentified. If you look at the script **3D_segment_extract_volume.py** there are few lines of code for image dilation/ erosion, and this is how some unidentified regions are automatically identified in the final segmented tif file. But sometime there might be errors, either caused by cellpose, or you just simply made mistakes while clicking patches. In that case, you can import your images and labels to ITK-SNAP for corrections.

ITK-SNAP is a free image annotation tool. Download [here](http://www.itksnap.org/pmwiki/pmwiki.php). I am using version 3.4 for this protocol but other versions should work similarly.

1. Situation: So after segmentations and all the steps above, you open the files in Amira, you found that some parts are ill-segmented:
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%206.png)
    
2. Open your raw image file and the segmentation file (*.tif) in image J
3. Save both of them as *.nrrd. Go to File > Save as > Nrrd.
4. Open the raw image nrrd in ITK-SNAP. In ITK-SNAP, Go to File > Open Main Image> Select your raw image nrrd file
5. Click [A] here to show the xy plane as full window.
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%207.png)
    
6. Go to Segmentation > Open Segmentation… > Select your segmentation nrrd file > Next and Finish.
7. Change tools to brush (1, see below), change active label to 255 (2), change brush style (3) and brush size (4), and annotate your ROI (5). Left click to draw and right click to erase
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%208.png)
    
8. Once you finished annotating all you wish to annotate, go to Segmentation > Save “…nrrd”
9. Now, open your new/ updated nrrd in image J. Convert it to tif there by File > Save as > *.tif
10. Now you can open it in Amira. See your correction?
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%209.png)
    

## Cellpose command line construction

Here I want to briefly go through how I find the command line for cellpose processing. The detailed explanation of each terms in the command line is given in the [cellpose documentation](https://cellpose.readthedocs.io/en/latest/command.html).

1. Here we are trying to perform cellpose segmentation on a 3D sample. 3D segmentation is complicated, therefore we will start with 2D segmentation. We will first slice the 3D tif to 2D image sequence, and put a few 2D images to cellpose, and try different AI models, different AI parameters and see what fit best with our need. Then we will use those paramaters to perform the 3D segmentation. Below are steps I followed to come to that line of command line in the beginning of this documentation. You could, simply follow what I wrote here but I highly recommand you to have a good read of the cellpose documentation.
2. Go to image J and open your 3D tif there. Go to File > Save as > Image sequence. And you can decide on where to save your 2D slices there.
3. Open Anaconda Prompt. Run `conda activate cellpose` to activate your cellpose environment, and run `python -m cellpose` to start cellpose GUI (If you dont know what I am doing, read the [cellpose documentation here](https://cellpose.readthedocs.io/en/latest/installation.html))
4. Open one of your image in cellpose. Try different parameters and see which one works well recognising most of your ROI. (Again, for how to use cellpose, see cellpose documentation) Have a look at [this page](https://cellpose.readthedocs.io/en/latest/command.html) for example command line.
5. So, I tried a few different parameters (And I tried a few different 2D slices), I found that model = cyto2 cell diameter = 75, flow threshold = 0.4, cellprobability = 0.0 works the best for me. And this is how I have `--pretrained_model cyto2` , `-diameter 75.0 --flow_threshold 0.4 --cellprob_threshold 0.0` in my command line. 
    
    Note that it will never be perfect. Thats why I design this pipeline to be half-human half-machine. So try play arround with different models and parameters, find what you think is “closer” to perfect.
    
    ![Untitled](Kidney%20segmentation%20in%203D%20a0dd5c532a394b2fb7aec3f703d79def/Untitled%2010.png)
    
6. Lets complete our command line. since we are dealing with a gray scale image with single channel, we have `--chan 0` . we want to accelerate the process by gpu. So there is `--use_gpu` . We would like to track our progress (as 3D segmentation can take awhile) so there is `--verbose` .
7. In cellpose, there are [two different methods for 3D segmentation](https://cellpose.readthedocs.io/en/latest/settings.html#d-settings). One is the “real” 3D segmentation, in which xy, yz, xz plane segmentation is performed, then we combine everything together. In that case we need `--do_3D` in our command line. However, our images have good xy resolution but worse z resolution. I think its not worth performing segmentation in z direction. So instead of —do_3D, I have `--stitch_threshold 0.1` here. This will help us to stitch the 2D planes together, therefore performing a like-3D segmentation for us. The default value for this is 0.1, however if its not doing the job correctly (e.g. two different ROIs are identified as one) then you might have to play around with this value a bit.
8. Finally, we need to tell where is our data allocated. For this one, we have `--dir U:\Kou\Vicky_kidney\original_vol\bin2_3` . And at the beginning we need `python -m cellpose`
9. So this is how our final executable command looks like: `python -m cellpose --dir U:\Kou\Vicky_kidney\original_vol\bin2_3 --pretrained_model cyto2 --chan 0 --diameter 75.0 --flow_threshold 0.4 --cellprob_threshold 0.0 --verbose --use_gpu --stitch_threshold 0.1`
