import argparse
import os.path as osp

from cityscapesscripts.preparation.json2labelImg import json2labelImg
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)

import os
from tqdm import tqdm

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Convert Cityscapes annotations to TrainIds')
#     parser.add_argument('cityscapes_path', help='cityscapes data path')
#     parser.add_argument('--gt-dir', default='gtFine', type=str)
#     parser.add_argument('-o', '--out-dir', help='output path')
#     parser.add_argument(
#         '--nproc', default=1, type=int, help='number of process')
#     args = parser.parse_args()
#     return args




def main():

    # os.system("rm /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/img_dir/test/*")
    # os.system("rm /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/img_dir/train/*")
    # os.system("rm /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/img_dir/val/*")

    # os.system("rm /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/ann_dir/test/*")
    # os.system("rm /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/ann_dir/train/*")
    # os.system("rm /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/ann_dir/val/*")

    data_root = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/"
    # 复制数据集到新位置
    with open(data_root+'train.txt') as f:
        for img_id in tqdm(f):
            img_id = img_id.strip()
            command = "cp /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/" + "hsi" + "/" + img_id + ".tiff " + data_root + "img_dir/train"
            # print(command)
            os.system(command)
            command = "cp /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/" + "label" + "/" + img_id + ".tiff " + data_root + "ann_dir/train"
            # print(command)
            os.system(command)

    with open(data_root+'val.txt') as f:
        for img_id in tqdm(f):
            img_id = img_id.strip()
            command = "cp /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/" + "hsi" + "/" + img_id + ".tiff " + data_root + "img_dir/val"
            # print(command)
            os.system(command)
            command = "cp /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/" + "label" + "/" + img_id + ".tiff " + data_root + "ann_dir/val"
            # print(command)
            os.system(command)

    with open(data_root+'test.txt') as f:
        for img_id in tqdm(f):
            img_id = img_id.strip()
            command = "cp /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/" + "hsi" + "/" + img_id + ".tiff " + data_root + "img_dir/test"
            # print(command)
            os.system(command)
            command = "cp /home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/" + "label" + "/" + img_id + ".tiff " + data_root + "ann_dir/test"
            # print(command)
            os.system(command)
    
    return 0


if __name__ == '__main__':
    main()
