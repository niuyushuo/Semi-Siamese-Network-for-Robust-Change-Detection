from PIL import Image

import numpy as np # linear algebra
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

IMG_FOLDER_NAME = "rgb"
IMG_POST_FOLDER_NAME = 'sar'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "mask"

#split = 'train'
split = 'val'
#split = 'test'

root_dir = '/home/yun13001/code/change_detection/whu_dataset/Unet_train/whu_dataset/'+split    #### train,val,test
#save_dir = '/home/yun13001/code/change_detection/whu_dataset/Unet_train/crop_img/'+split+'/rgb/' ##### rgb,sar,mask
#save_dir = '/home/yun13001/code/change_detection/whu_dataset/Unet_train/crop_img/'+split+'/sar/'
save_dir = '/home/yun13001/code/change_detection/whu_dataset/Unet_train/crop_img/'+split+'/mask/'

img_size=256
crop_size=32
ratio=int(256/32)

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name+'.tif')

def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name+'.png')

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name+'.png')

list_path = os.path.join(root_dir, LIST_FOLDER_NAME, split+'.txt')
img_name_list = load_img_name_list(list_path)

print(len(img_name_list))
#print(img_name_list)

rgb=[]
sar=[]
mask=[]
index=0

for i in range(len(img_name_list)):
    A_path=get_img_path(root_dir, img_name_list[i]) 
    img = Image.open(A_path)

    for m in range(ratio):
        for n in range(ratio):

            box = (m*crop_size, n*crop_size, (m+1)*crop_size, (n+1)*crop_size)

            output_img=img.crop(box)

            output_img.save(save_dir+str(index) +".png")
            index = index +1
                   
    

