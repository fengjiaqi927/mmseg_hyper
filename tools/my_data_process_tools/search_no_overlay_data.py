import os
import mmcv
import cv2
from tqdm import tqdm
import tifffile as tiff
import json

from functools import partial, reduce
from multiprocessing import Manager, Pool
from mmengine import print_log

import numpy as np

def search_neighbor(id_now,img_id_list,searched_neighbord,label_img_dir, lock, prog,
                 total):
    img = tiff.imread(label_img_dir+id_now+".tiff")
    h, w, _ = img.shape
    img_A = img[:int(h/2),:int(w/2),0]
    img_B = img[:int(h/2),int(w/2):,0]
    img_C = img[int(h/2):,int(w/2):,0]
    img_D = img[int(h/2):,:int(w/2),0]
    for img_id in tqdm(img_id_list):   
        img_now = tiff.imread(label_img_dir+img_id+".tiff")
        img_now_A = img_now[:int(h/2),:int(w/2),0]
        img_now_B = img_now[:int(h/2),int(w/2):,0]
        img_now_C = img_now[int(h/2):,int(w/2):,0]
        img_now_D = img_now[int(h/2):,:int(w/2),0]
        if searched_neighbord[img_id][0] == -1:
            if cv2.countNonZero(cv2.absdiff(img_now_A, img_B)) == 0:
                searched_neighbord[img_id][0] = id_now
                searched_neighbord[id_now][1] = img_id
                searched_neighbord[img_id][2] = id_now
                searched_neighbord[id_now][3] = img_id
            elif cv2.countNonZero(cv2.absdiff(img_now_A, img_C)) == 0:
                searched_neighbord[img_id][0] = id_now
                searched_neighbord[img_id][1] = id_now
                searched_neighbord[id_now][2] = img_id
                searched_neighbord[id_now][3] = img_id
            elif cv2.countNonZero(cv2.absdiff(img_now_A, img_D)) == 0:
                searched_neighbord[img_id][0] = id_now
                searched_neighbord[id_now][3] = img_id
        if searched_neighbord[img_id][1] == -1:
            if cv2.countNonZero(cv2.absdiff(img_now_B, img_A)) == 0:
                searched_neighbord[id_now][0] = img_id
                searched_neighbord[img_id][1] = id_now
                searched_neighbord[id_now][2] = img_id
                searched_neighbord[img_id][3] = id_now
            elif cv2.countNonZero(cv2.absdiff(img_now_B, img_C)) == 0:
                searched_neighbord[img_id][1] = id_now
                searched_neighbord[id_now][2] = img_id
            elif cv2.countNonZero(cv2.absdiff(img_now_B, img_D)) == 0: 
                searched_neighbord[img_id][0] = id_now
                searched_neighbord[img_id][1] = id_now
                searched_neighbord[id_now][2] = img_id
                searched_neighbord[id_now][3] = img_id
        if searched_neighbord[img_id][2] == -1:
            if cv2.countNonZero(cv2.absdiff(img_now_C, img_A)) == 0:
                searched_neighbord[id_now][0] = img_id
                searched_neighbord[id_now][1] = img_id
                searched_neighbord[img_id][2] = id_now
                searched_neighbord[img_id][3] = id_now
            elif cv2.countNonZero(cv2.absdiff(img_now_C, img_B)) == 0:
                searched_neighbord[id_now][1] = img_id
                searched_neighbord[img_id][2] = id_now
            elif cv2.countNonZero(cv2.absdiff(img_now_C, img_D)) == 0:
                searched_neighbord[img_id][0] = id_now
                searched_neighbord[id_now][1] = img_id
                searched_neighbord[img_id][2] = id_now
                searched_neighbord[id_now][3] = img_id
        if searched_neighbord[img_id][3] == -1:
            if cv2.countNonZero(cv2.absdiff(img_now_D, img_A)) == 0:
                searched_neighbord[id_now][0] = img_id
                searched_neighbord[img_id][3] = id_now
            elif cv2.countNonZero(cv2.absdiff(img_now_D, img_B)) == 0:
                searched_neighbord[id_now][0] = img_id
                searched_neighbord[id_now][1] = img_id
                searched_neighbord[img_id][2] = id_now
                searched_neighbord[img_id][3] = id_now
            elif cv2.countNonZero(cv2.absdiff(img_now_D, img_C)) == 0:
                searched_neighbord[id_now][0] = img_id
                searched_neighbord[img_id][1] = id_now
                searched_neighbord[id_now][2] = img_id
                searched_neighbord[img_id][3] = id_now
    lock.acquire()
    prog.value += 1
    msg = f'({prog.value / total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {id_now}"
    print_log(msg, 'current')
    lock.release()
        

def search_neighbor_8(id_now,img_id_list,searched_neighbord,label_img_dir, lock, prog,
                 total):
    img = tiff.imread(label_img_dir+id_now+".tiff")
    h, w, _ = img.shape
    """
        [a ,b
         c, d]
    """
    img_A = img[:int(h/2),:int(w/2),0]
    img_B = img[:int(h/2),int(w/2):,0]
    img_C = img[int(h/2):,:int(w/2),0]
    img_D = img[int(h/2):,int(w/2):,0]
    for img_id in tqdm(img_id_list):   
        img_now = tiff.imread(label_img_dir+img_id+".tiff")
        img_now_A = img_now[:int(h/2),:int(w/2),0]
        img_now_B = img_now[:int(h/2),int(w/2):,0]
        img_now_C = img_now[int(h/2):,:int(w/2),0]
        img_now_D = img_now[int(h/2):,int(w/2):,0]# 左上0，上1，右上2，左3，右4，左下5，下6，右下7
        if searched_neighbord[img_id][0] == -1: # 左上
            if cv2.countNonZero(cv2.absdiff(img_now_A, img_D)) == 0:
                searched_neighbord[img_id][0] = int(id_now)
                searched_neighbord[id_now][7] = int(img_id)
                continue
        if searched_neighbord[img_id][1] == -1:# 上
            if cv2.countNonZero(cv2.absdiff(img_now_A, img_C)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_B, img_D)) == 0:
                searched_neighbord[img_id][1] = int(id_now)
                searched_neighbord[id_now][6] = int(img_id)
                continue
        if searched_neighbord[img_id][2] == -1:# 右上
            if cv2.countNonZero(cv2.absdiff(img_now_B, img_C)) == 0:
                searched_neighbord[img_id][2] = int(id_now)
                searched_neighbord[id_now][5] = int(img_id)
                continue
        if searched_neighbord[img_id][3] == -1:# 左
            if cv2.countNonZero(cv2.absdiff(img_now_A, img_B)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_C, img_D)) == 0:
                searched_neighbord[img_id][3] = int(id_now)
                searched_neighbord[id_now][4] = int(img_id)
                continue
        if searched_neighbord[img_id][4] == -1:# 右
            if cv2.countNonZero(cv2.absdiff(img_now_B, img_A)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_D, img_C)) == 0:
                searched_neighbord[img_id][4] = int(id_now)
                searched_neighbord[id_now][3] = int(img_id)
                continue   
        if searched_neighbord[img_id][5] == -1:# 左下
            if cv2.countNonZero(cv2.absdiff(img_now_C, img_B)) == 0 :
                searched_neighbord[img_id][5] = int(id_now)# 左上0，上1，右上2，左3，右4，左下5，下6，右下7
                searched_neighbord[id_now][2] = int(img_id)
                continue
        if searched_neighbord[img_id][6] == -1:# 下
            if cv2.countNonZero(cv2.absdiff(img_now_C, img_A)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_D, img_B)) == 0:
                searched_neighbord[img_id][6] = int(id_now)
                searched_neighbord[id_now][1] = int(img_id)
                continue
        if searched_neighbord[img_id][7] == -1:# 右下
            if cv2.countNonZero(cv2.absdiff(img_now_D, img_A)) == 0 :
                searched_neighbord[img_id][7] = int(id_now)
                searched_neighbord[id_now][0] = int(img_id)
                continue
    lock.acquire()
    prog.value += 1
    msg = f'({prog.value / total:3.1%} {prog.value}:{total})'
    msg += ' - ' + f"Filename: {id_now}"
    print_log(msg, 'current')
    lock.release()
    print(searched_neighbord[id_now])

    # return searched_neighbord

# quanju_dict = {}

# def search_neighbor_8_new(id_now,img_id_list,searched_neighbord,label_img_dir, lock, prog,
#                  total):
#     img = tiff.imread(label_img_dir+id_now+".tiff")
#     h, w, _ = img.shape
#     img_A = img[:int(h/2),:int(w/2),0]
#     img_B = img[:int(h/2),int(w/2):,0]
#     img_C = img[int(h/2):,int(w/2):,0]
#     img_D = img[int(h/2):,:int(w/2),0]
#     for img_id in tqdm(img_id_list):   
#         img_now = tiff.imread(label_img_dir+img_id+".tiff")
#         img_now_A = img_now[:int(h/2),:int(w/2),0]
#         img_now_B = img_now[:int(h/2),int(w/2):,0]
#         img_now_C = img_now[int(h/2):,int(w/2):,0]
#         img_now_D = img_now[int(h/2):,:int(w/2),0]# 左上0，上1，右上2，左3，右4，左下5，下6，右下7
#         if searched_neighbord[img_id][0] == -1: # 左上
#             if cv2.countNonZero(cv2.absdiff(img_now_A, img_D)) == 0:
#                 searched_neighbord[img_id][0] = int(id_now)
#                 searched_neighbord[id_now][7] = int(img_id)
#                 continue
#         if searched_neighbord[img_id][1] == -1:# 上
#             if cv2.countNonZero(cv2.absdiff(img_now_A, img_C)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_B, img_D)) == 0:
#                 searched_neighbord[img_id][1] = int(id_now)
#                 searched_neighbord[id_now][6] = int(img_id)
#                 continue
#         if searched_neighbord[img_id][2] == -1:# 右上
#             if cv2.countNonZero(cv2.absdiff(img_now_B, img_C)) == 0:
#                 searched_neighbord[img_id][2] = int(id_now)
#                 searched_neighbord[id_now][5] = int(img_id)
#                 continue
#         if searched_neighbord[img_id][3] == -1:# 左
#             if cv2.countNonZero(cv2.absdiff(img_now_A, img_B)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_C, img_D)) == 0:
#                 searched_neighbord[img_id][3] = int(id_now)
#                 searched_neighbord[id_now][4] = int(img_id)
#                 continue
#         if searched_neighbord[img_id][4] == -1:# 右
#             if cv2.countNonZero(cv2.absdiff(img_now_B, img_A)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_D, img_C)) == 0:
#                 searched_neighbord[img_id][4] = int(id_now)
#                 searched_neighbord[id_now][3] = int(img_id)
#                 continue   
#         if searched_neighbord[img_id][5] == -1:# 左下
#             if cv2.countNonZero(cv2.absdiff(img_now_C, img_B)) == 0 :
#                 searched_neighbord[img_id][5] = int(id_now)# 左上0，上1，右上2，左3，右4，左下5，下6，右下7
#                 searched_neighbord[id_now][2] = int(img_id)
#                 continue
#         if searched_neighbord[img_id][6] == -1:# 下
#             if cv2.countNonZero(cv2.absdiff(img_now_C, img_A)) == 0 & cv2.countNonZero(cv2.absdiff(img_now_D, img_B)) == 0:
#                 searched_neighbord[img_id][6] = int(id_now)
#                 searched_neighbord[id_now][1] = int(img_id)
#                 continue
#         if searched_neighbord[img_id][7] == -1:# 右下
#             if cv2.countNonZero(cv2.absdiff(img_now_D, img_A)) == 0 :
#                 searched_neighbord[img_id][7] = int(id_now)
#                 searched_neighbord[id_now][0] = int(img_id)
#                 continue
#     lock.acquire()
#     prog.value += 1
#     msg = f'({prog.value / total:3.1%} {prog.value}:{total})'
#     msg += ' - ' + f"Filename: {id_now}"
#     print_log(msg, 'current')
#     lock.release()





if __name__ == "__main__":
    
    # 寻找图片之间的位置关系

    label_img_dir = "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/hsi/"
    img = tiff.imread(label_img_dir+'1'+".tiff")
    print(img.shape)
    print(0)
    # img_id_list = [file_name.split(".")[0] for file_name in os.listdir(label_img_dir) if file_name.endswith(".tiff")]
    # searched_neighbord = {
    #     img_id_list[i]:[-1,-1,-1,-1,-1,-1,-1,-1]# 左上0，上1，右上2，左3，右4，左下5，下6，右下7
    #     for i in range(len(img_id_list))
    # }

    # nproc = 1
    # manager = Manager()

    # _load_func = partial(
    #         search_neighbor_8,
    #         img_id_list=img_id_list,
    #         searched_neighbord=searched_neighbord,
    #         label_img_dir=label_img_dir,
    #         lock=manager.Lock(),
    #         prog=manager.Value('i', 0),
    #         total=len(img_id_list)
    #         )
    # if nproc > 1:
    #     pool = Pool(nproc)
    #     contents = pool.map(_load_func, img_id_list)
    #     pool.close()
    # else:
    #     contents = list(map(_load_func, img_id_list))


    # # for img in tqdm(img_id_list):
    # #     search_neighbor(img,img_id_list,searched_neighbord,label_img_dir)
    
    # file_name = 'searched_neighbord.json'
    # with open(file_name, 'w') as json_file:
    #     json.dump(searched_neighbord, json_file)                

    # 根据图片位置获得不相交的图片集
    file_name = 'searched_neighbord.json'
    with open(file_name, 'r') as json_file:
        loaded_dict = json.load(json_file)
    
    top_left_list = [id for id in loaded_dict if loaded_dict[id][0] == -1 & loaded_dict[id][1] == -1 & loaded_dict[id][2] == -1 & loaded_dict[id][3] == -1 & loaded_dict[id][5] == -1]

    # [id for id in loaded_dict if loaded_dict[id][0] == -1 & 
    #                              loaded_dict[id][1] == -1 & 
    #                              loaded_dict[id][2] == -1 & 
    #                              loaded_dict[id][3] == -1 & 
    #                              loaded_dict[id][4] == -1 & 
    #                              loaded_dict[id][5] == -1 &
    #                              loaded_dict[id][6] == -1 &
    #                              loaded_dict[id][7] == -1 
    #                              ]
    
    # [id for id in loaded_dict if loaded_dict[id][0] != -1 & 
    #                              loaded_dict[id][1] != -1 & 
    #                              loaded_dict[id][2] != -1 & 
    #                              loaded_dict[id][3] != -1 & 
    #                              loaded_dict[id][4] != -1 & 
    #                              loaded_dict[id][5] != -1 &
    #                              loaded_dict[id][6] != -1 &
    #                              loaded_dict[id][7] != -1 
    #                              ]


    array_dict = {}
    for top_left in top_left_list:
        temp_id = top_left
        all_list = []
        while ~(loaded_dict[temp_id][4]==-1 & loaded_dict[temp_id][6] == -1):
            temp_id_row = temp_id
            row_list = [temp_id_row]
            while loaded_dict[temp_id_row][4]!=-1:
                temp_id_row = str(loaded_dict[temp_id_row][4])
                row_list.append(temp_id_row)
            all_list.append(row_list)
            temp_id = str(loaded_dict[temp_id][6])
            if temp_id == '-1':
                break
        array_dict[top_left] = all_list
    
    print(0)
    # total_elements = sum(len(list_temp) for list_temp in array_dict['2811'])
    # 5742    6968
    # 1391    67
    # 2811    104
    # 6203    1
    # total   7140
    img_list_pick_up = []

    # for top_left in array_dict[:1]:
    print('5742')
    temp_list = array_dict['5742']
    temp_array = np.array(temp_list)
    img_list_pick_up.append(temp_array[::2,::2].flatten().tolist())

    keys = list(array_dict.keys())
    for top_left in keys[1:]:
        print(top_left)
        # temp_list = array_dict[top_left] if top_left!=keys[1] else array_dict[top_left][0]
        temp_list = [item for sublist in array_dict[top_left] for item in sublist] 
        temp_list = temp_list[::2]
        img_list_pick_up.append(temp_list)

    print("#################################")
    flattened_list = [item for sublist in img_list_pick_up for item in sublist]

    # 检查1855张图像中有无重合部分
    # 1768+34+52+1
    # for temp in tqdm(flattened_list):
    #     for img_id in list(loaded_dict.keys()):
    #         if int(temp) in loaded_dict[img_id] and img_id in flattened_list:
    #             print(temp)
    
    # 随机划分训练验证测试集
    # 训练1115，验证185，测试555
    import random

    random_seed = 7
    random.seed(random_seed)

    random.shuffle(flattened_list)

    train_list = flattened_list[:1115]
    val_list = flattened_list[1115:1115+185]
    test_list = flattened_list[-555:]
    # "/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train/msi/"
    with open("/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/train.txt", "w") as file:
        for item in train_list:
            file.write("%s\n" % item)
    with open("/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/val.txt", "w") as file:
        for item in val_list:
            file.write("%s\n" % item)
    with open("/home/fengjq/3grade/C2Seg/data/C2Seg_BW/train_val_test/test.txt", "w") as file:
        for item in test_list:
            file.write("%s\n" % item)




    

