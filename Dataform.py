# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import h5py

class Mydatasetmat(Data.Dataset):

    def __init__(self, csv_file1, feature):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = h5py.File(csv_file1,'r')
        self.data = self.data['data'][:]

        self.data = torch.from_numpy(self.data)
        self.data = self.data.permute(2, 3, 0, 1)
        self.feature = -feature


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        x = self.data[index,:self.feature,:,:]
        y = self.data[index,self.feature:,:,:]
        data1 = x
        item_A = data1
        data2 = y[:,1,1]
        item_B = data2

        return {'A': item_A, 'B': item_B}
