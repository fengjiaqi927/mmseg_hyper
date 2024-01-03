import numpy as np
import tifffile as tiff
from tqdm import tqdm

if __name__ == "__main__":
    data_root = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/"
    total_img_id_list = []
    train_list = []
    val_list = []
    test_list = []
    with open(data_root+'train.txt') as f:
        for img_id in f:
            img_id = img_id.strip()
            total_img_id_list.append(img_id)
            train_list.append(img_id)
    with open(data_root+'val.txt') as f:
        for img_id in f:
            img_id = img_id.strip()
            total_img_id_list.append(img_id)
            val_list.append(img_id)
    with open(data_root+'test.txt') as f:
        for img_id in f:
            img_id = img_id.strip()
            total_img_id_list.append(img_id) 
            test_list.append(img_id)
    img_dirs = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/label/"
    img_gt = tiff.imread(img_dirs + "1.tiff")
    print(0)
    category = np.zeros(14)

    for img_id in tqdm(total_img_id_list):
        img_dir = img_dirs + img_id + ".tiff"
        img = tiff.imread(img_dir)
        for d in range(14):
            category[d] += len(np.where(img==d)[0])
            

    # 统计训练集，验证集 和 测试集的类别情况，数据集切分结果写出到txt，数据统计结果写出到windows的excel
    category_train = np.zeros(14)
    category_val = np.zeros(14)
    category_test = np.zeros(14)
    for img_id in tqdm(train_list):
        img_dir = img_dirs + img_id + ".tiff"
        img = tiff.imread(img_dir)
        for d in range(14):
            category_train[d] += len(np.where(img==d)[0])

    for img_id in tqdm(val_list):
        img_dir = img_dirs + img_id + ".tiff"
        img = tiff.imread(img_dir)
        for d in range(14):
            category_val[d] += len(np.where(img==d)[0])

    for img_id in tqdm(test_list):
        img_dir = img_dirs + img_id + ".tiff"
        img = tiff.imread(img_dir)
        for d in range(14):
            category_test[d] += len(np.where(img==d)[0])


    for i in category_train:
        print(i/np.sum(category_train))
    print("##############################")
    for i in category_val:
        print(i/np.sum(category_val))
    print("##############################")
    for i in category_test:
        print(i/np.sum(category_test))
    print("##############################")
    for i in category:
        print(i/np.sum(category))
