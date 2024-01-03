''''
    让样本分布在训练，验证，测试集合中保持一致
    函数步骤：
    统计所有数据的类别像素数量
    按照像素数对类别进行排序，从最低到最高
    得到一个dict["类别"] = [img_id_1,...,img_id_k]
    筛选一遍得到一个无交集的子集，求和为全体数据集
    对该字典开始遍历，按照6：1：3设置训练，验证，测试集
    分别统计训练，验证，测试集的样本分布，查看是否相似，且与总体分布相似
'''

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
    img_dirs = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/label/"
    img_gt = tiff.imread(img_dirs + "1.tiff")
    print(0)
    category = np.zeros(14)
    category_img_dict = {
        i:[]
        for i in range(14)
    }
    for img_id in tqdm(total_img_id_list):
        img_dir = img_dirs + img_id + ".tiff"
        img = tiff.imread(img_dir)
        for d in range(14):
            category[d] += len(np.where(img==d)[0])
            if len(np.where(img==d)[0])!=0:
                category_img_dict[d].append(img_id)
    sorted_category_img_dict = dict(sorted(category_img_dict.items(), key=lambda x: len(x[1])))
    def pick_up_repeat(temp_dict,i=1):
        if i < len(temp_dict.keys()):
            list_1 = [temp_dict[list(sorted_category_img_dict.keys())[a]] for a in range (i)]
            list_1 = [item for sublist in list_1 for item in sublist]
            temp_list = []
            for item in temp_dict[list(sorted_category_img_dict.keys())[i]]:
                if item not in list_1:
                    temp_list.append(item)
            temp_dict[list(sorted_category_img_dict.keys())[i]] = temp_list
            pick_up_repeat(temp_dict,i+1)

    pick_up_repeat(sorted_category_img_dict)

    sum_num = 0
    import random
    random_seed = 14
    random.seed(random_seed)
    train_list, val_list, test_list = [],[],[]
    for i in sorted_category_img_dict:
        total = len(sorted_category_img_dict[i])
        random.shuffle(sorted_category_img_dict[i])
        train = int(total*0.6)
        val = int(total*0.1)
        test = total - train - val
        print("{},{},{},{},{}".format(i,train,val,test,total))
        train_list += sorted_category_img_dict[i][:train]
        val_list += sorted_category_img_dict[i][train:train+val]
        test_list += sorted_category_img_dict[i][-test:]
        sum_num += total
    print(sum_num)

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


    with open("/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test_balance/train.txt", "w") as file:
        for item in train_list:
            file.write("%s\n" % item)
    with open("/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test_balance/val.txt", "w") as file:
        for item in val_list:
            file.write("%s\n" % item)
    with open("/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test_balance/test.txt", "w") as file:
        for item in test_list:
            file.write("%s\n" % item)

    print(0)

