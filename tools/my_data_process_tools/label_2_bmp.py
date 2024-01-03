import tifffile as tiff
from tqdm import tqdm
import numpy as np
import cv2

# total_img_id_list = 

# for img_id in tqdm(total_img_id_list):
#     img_dir = img_dirs + img_id + ".tiff"
img_dir = "tools/my_data_process_tools/img_dir/C2Seg_BW_train_label_0.tiff"
output_path = "tools/my_data_process_tools/result_dir/"

classes=['Background','Surface_water','Street','Urban_Fabric',
                 'Industrial_commercial_and_transport','Mine_dump_and_construction_sites',
                 'Artificial_vegetated_areas','Arable_Land','Permanent_Crops',
                 'Pastures','Forests','Shrub','Open_spaces_with_no_vegetation','Inland_wetlands']

img_gt = tiff.imread(img_dir)
unique_values =  np.unique(img_gt)
for value in unique_values:
    binary_mask = np.where(img_gt == value, 255, 0).astype(np.uint8)
    filename = f"mask_{value}_" +classes[value] + ".png"
    cv2.imwrite(output_path + filename, binary_mask)