import os
import glob
import random

from torch.utils.data import Dataset
import torch
import numpy as np
import SimpleITK as sitk


class MySet(Dataset):
    def __init__(self, data_list, segment):
        self.data_list = data_list
        self.segment = segment

    def __getitem__(self, item):
        data_dict = self.data_list[item]
        data_path = data_dict["data"]
        mask_path = data_dict["label"]

        data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        data = np.transpose(data, (2,0,1))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = np.transpose(mask, (2,0,1))
        if self.segment == 0:
            mask = np.where(mask>0, 1, 0)
        elif self.segment == 1:
            mask = np.where(mask>1, 1, 0)
        data = self.normalize(data)
        data = data[np.newaxis, :, :, :]
        mask = mask.astype(np.float32)
        mask = mask[np.newaxis, :, :, :]
        width = 30
        x = mask.shape[1]
        if x < width:
            data = np.resize(data, (1,width,data.shape[2],data.shape[3]))
            mask = np.resize(data, (1,width,mask.shape[2],mask.shape[3]))
        mask_tensor = torch.from_numpy(mask)
        data_tensor = torch.from_numpy(data)

        return data_tensor, mask_tensor

    @staticmethod
    def normalize(data):
        data = data.astype(np.float32)
        data = (data - np.min(data))/(np.max(data) - np.min(data))
        return data

    def __len__(self):
        return len(self.data_list)


def create_list(data_path, ratio=0.8):

    label_name = 'segmentation.nii.gz'
    data_name = 'imaging.nii.gz'
    list_all = [{'data': os.path.join("/content/kits19/data/case_{0:5d}".format(i), data_name), 'label': os.path.join("/content/kits19/data/case_{0:5d}".format(i), label_name)} for i in range(200)]

    cut = int(ratio * len(list_all))
    train_list = list_all[:cut]
    test_list = list_all[cut:]

    random.shuffle(train_list)

    return train_list, test_list
