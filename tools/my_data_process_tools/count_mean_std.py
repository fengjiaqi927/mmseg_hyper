import numpy as np
import tifffile as tiff
from tqdm import tqdm

if __name__ == "__main__":
    data_root = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/"
    total_img_id_list = []
    with open(data_root+'train.txt') as f:
        for img_id in f:
            img_id = img_id.strip()
            total_img_id_list.append(img_id)
    with open(data_root+'val.txt') as f:
        for img_id in f:
            img_id = img_id.strip()
            total_img_id_list.append(img_id)
    with open(data_root+'test.txt') as f:
        for img_id in f:
            img_id = img_id.strip()
            total_img_id_list.append(img_id)
    print(0)
    img_dirs = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/hsi/"
    h,w,c = tiff.imread(img_dirs + "1.tiff").shape

    mean = np.zeros(c)
    std = np.zeros(c)
    
    for img_id in tqdm(total_img_id_list):
        img_dir = img_dirs + img_id + ".tiff"
        img = tiff.imread(img_dir)
        for d in range(c):
            mean[d] += img[:,:,d].mean()
            std[d] += img[:,:,d].std()
    mean = mean/(len(total_img_id_list))
    std = std/(len(total_img_id_list))
    np.save('mean.npy', mean)
    np.save('std.npy', std)

    mean = np.load('mean.npy')
    std = np.load('std.npy')
    print("%3f{}".format(mean))
    



