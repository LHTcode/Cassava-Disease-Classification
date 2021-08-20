import PIL.ImageShow
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
from PIL import Image
import csv
from torch.utils.data import DataLoader
from torchvision import transforms


class DataSet(Dataset):
    def __init__(self,path,train=True,extraimages=False,transform=None):
        super(DataSet,self).__init__()
        train_path = os.path.join(path,'cassava-disease','train')           #这个‘0’目录有啥意义我也不清楚，数据集下载下来后就有的
        test_path = os.path.join(path,'cassava-disease','test','0')
        labels_path = os.path.join(path,'cassava-disease','sample_submission_file.csv')
        extraimages_path = os.path.join(path,'cassava-disease','extraimages')
        self.transform = transform
        self.imgs_PIL = []                #存储训练或测试图片
        self.extraimages_PIL = []         #存储所有额外训练图片，格式为PIL
        self.img_labels = []              #存储训练或测试图片的标签

        #获取所有训练图片,并获取train_label
        if train == True:
            cbb_num = 0
            cbsd_num = 0
            cgm_num = 0
            cmd_num = 0
            healthy_num = 0
            folder_list = os.listdir(train_path)                                    #将train目录下的文件夹名存储起来下面循环取出
            for folder in folder_list:
                folder_path = os.path.join(train_path,folder)
                img_num = 0
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('jpg'):
                        self.imgs_PIL.append(Image.open(os.path.join(folder_path,img_name)))    #这一步取出了所有图片
                        img_num += 1
                if folder == 'cbb':
                    cbb_num = img_num
                if folder == 'cbsd':
                    cbsd_num = img_num
                if folder == 'cgm':
                    cgm_num = img_num
                if folder == 'cmd':
                    cmd_num = img_num
                if folder == 'healthy':
                    healthy_num = img_num
            for i in range(cbb_num):self.img_labels.append(0)
            for i in range(cbsd_num):self.img_labels.append(1)
            for i in range(cgm_num):self.img_labels.append(2)
            for i in range(cmd_num):self.img_labels.append(3)
            for i in range(healthy_num):self.img_labels.append(4)

        #获取测试图片和labels
        if train == False:
            #获取所有的测试图片
            for img_name in os.listdir(test_path):
                if img_name.endswith('jpg'):
                    self.imgs_PIL.append(Image.open(os.path.join(test_path, img_name)))  # 这一步取出了所有图片

            #获取test_label
            with open(labels_path) as f:
                reader = csv.reader(f)
                reader.__next__()
                for labels in reader:
                    if labels[0] == 'cbb':
                        self.img_labels.append(0)
                    if labels[0] == 'cbsd':
                        self.img_labels.append(1)
                    if labels[0] == 'cgm':
                        self.img_labels.append(2)
                    if labels[0] == 'cmd':
                        self.img_labels.append(3)
                    if labels[0] == 'healthy':
                        self.img_labels.append(4)

        # 获取额外图片
        if extraimages == True:
            folder_list = os.listdir(extraimages_path)  # 将extraimages目录下的文件夹名存储起来下面循环取出
            for folder in folder_list:
                folder_path = os.path.join(extraimages_path, folder)
                for img_name in os.listdir(folder_path):
                    if img_name.endswith('jpg'):
                        self.extraimages_PIL.append(Image.open(os.path.join(folder_path, img_name)))  # 这一步取出了所有图片



    def __getitem__(self, index):
        if self.transform != None:
            self.imgs_PIL[index] = self.transform(self.imgs_PIL[index])
            #统一图片size
            resize = (600,600)                      #由于图片尺寸不一的话Dataloader不接受我的数据，所以这里做一步统一尺寸的操作
            transrom = transforms.Resize(resize)
            self.imgs_PIL[index] = transrom(self.imgs_PIL[index])

        return self.imgs_PIL[index],self.img_labels[index]
    def __len__(self):

        return len(self.img_labels)


if __name__ == '__main__':
    data_set = DataSet(path='data_set',train=True,transform=ToTensor())
    dataloader = DataLoader(data_set,batch_size=1)
    for data in dataloader:
        img,label = data
        # print(img.size(),label)