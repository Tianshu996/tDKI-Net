#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
from net import tEncoder
from net import Mappingt
from net import qEncoders
from Dataform import Mydatasetmat
import argparse
import os
import sys
import time
import nibabel as nib

a = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16384, help='size of the batches') 
parser.add_argument('--input_nc', type=int, default=150, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=7, help='number of channels of output data')
parser.add_argument('--size', type=int, default=3, help='size of the data (squared assumed)')
parser.add_argument('--cuda', default='True',action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--tsp', type=str, default='./output/tsp.pth',help='sp checkpoint file')
parser.add_argument('--Tmapping', type=str, default='./output/Tmapping.pth',help='tmapping checkpoint file')
parser.add_argument('--Kmapping', type=str, default='./output/Kmapping.pth',help='kmapping checkpoint file')
parser.add_argument('--dataset', type=str, default='./test.mat')
parser.add_argument('--mask', type=str, default='./mask0.nii')


opt = parser.parse_args()
print(opt)
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


tsp = tEncoder(feat=opt.output_nc-2)
Tmapping = Mappingt()
Kmapping = qEncoders(301, 75, opt.input_nc, opt.output_nc-2)

if opt.cuda:
    tsp.cuda()
    Tmapping.cuda()
    Kmapping.cuda()

tsp.load_state_dict(torch.load(opt.tsp))
Tmapping.load_state_dict(torch.load(opt.Tmapping))
Kmapping.load_state_dict(torch.load(opt.Kmapping))

# Set model's test mode
tsp.eval()
Tmapping.eval()
Kmapping.eval()

Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc)

# Dataset loader
testset = Mydatasetmat(opt.dataset, opt.output_nc)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=opt.batchSize,
    shuffle=False,
    drop_last=True,
)
###################################

###### Testing######

# Create output dirs if they don't exist


store = []
store2 = []
store3 = []
store4 = []
with torch.no_grad():
    for i, batch in enumerate(testloader):
        # Set model input
        real_A = (Variable(input_A.copy_(batch['A'])))
        real_B = Variable(input_B.copy_(batch['B']))

        real_BDK = real_B[:, :-2]
        real_BK = real_B[:, -2:]

        DKI = Kmapping(real_A)

        t = tsp(DKI)

        results = Tmapping(t)

        output = results.detach().cpu().numpy()
        result1 = np.array(output)
        store.append(result1)
        result1 = np.array(store)
        result1 = np.reshape(result1, (result1.shape[0] * result1.shape[1], 2))

        result1[:,0] = result1[:,0] / 10
        result1[:,1] = result1[:,1] * 100

        output2 = real_BK.detach().cpu().numpy()
        result2 = np.array(output2)
        store2.append(result2)
        result2 = np.array(store2)
        result2 = np.reshape(result2, (result2.shape[0] * result2.shape[1], 2))

        output3 = DKI.detach().cpu().numpy()
        result3 = np.array(output3)
        store3.append(result3)
        result3 = np.array(store3)
        result3 = np.resize(result3, (result3.shape[0] * result3.shape[1], 5))

        output4 = real_BDK.detach().cpu().numpy()
        result4 = np.array(output4)
        store4.append(result4)
        result4 = np.array(store4)
        result4 = np.resize(result4, (result4.shape[0] * result4.shape[1], 5))

    mask_nii = nib.load(opt.mask)

    mask = mask_nii.get_fdata()

    affine = mask_nii.affine

    for ii in range(2):

        data0 = result1[:,ii].reshape(mask.shape,order='F') * mask

        image0 = nib.Nifti1Image(data0, affine)

        nib.save(image0,'./output/resultKarger'+str(ii)+'.nii.gz')

        data1 = result2[:,ii].reshape(mask.shape,order='F') * mask

        image1 = nib.Nifti1Image(data1, affine)

        nib.save(image1,'./output/labelKarger'+str(ii)+'.nii.gz')

