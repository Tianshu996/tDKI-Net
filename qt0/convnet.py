import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as Data
import scipy.io as scio
# import pandas as pd
import numpy as np
import pdb
import scipy.io as io



class Generator1(nn.Module):
    def __init__(self,feat):
        super(Generator1, self).__init__()
        input = 256

        self.dcblock0 = nn.Sequential(
            nn.Linear(input,input,bias=True),

        )

        self.dcblock1 = nn.Sequential(
            nn.Linear(input,input,bias=True),

        )



        self.w0block = nn.Sequential(
            nn.Linear(feat,input,bias=True),
        )

        self.w1block = nn.Sequential(
            nn.Linear(input, input, bias=True),

        )

        self.activ0 = nn.Sequential(
            nn.Threshold(0.001, 0, inplace=False),
        )






    def forward(self,y):
        # x = self.c1(x)
        wy1 = self.w0block(y)
        wy2 = self.w1block(wy1)
        x2 = wy1




        for i in range (10):
            x1 = self.activ0(x2)
            x1 = self.dcblock0(x1) + wy1
            x20 = self.activ0(x1)
            x2 = self.dcblock1(x20) + x2 + wy2
            #
            # x1 = self.dcblock0(wy1) + wy1
            # x1 = self.dcblock1(x1) + wy2

        y1 = self.activ0(x2)

        return y1












class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        input=256
        model = [
            # nn.GELU(),
            # nn.Linear(300,1,bias=True),
            # nn.Sigmoid(),
            # nn.Threshold(0.0001, 0, inplace=True),

            # nn.Conv2d(in_channels=input,out_channels=2*input,kernel_size=3),
            nn.Linear(input, 2 * input, bias=True),

            nn.GELU(),
            # nn.Dropout(p=0.2),
            nn.Linear(2 * input, 4 * input, bias=True),
            # nn.Dropout(p=0.2),
            nn.GELU(),

            nn.Linear(4 * input, 2 * input, bias=True),
            nn.GELU(),
            nn.Linear(2 * input, 2 , bias=True),


            nn.ReLU(),
            # nn.Threshold(0.001,0)
        ]

        self.model = nn.Sequential(*model)



    def forward(self,x):
        x1 = self.model(x)
        # x2 = self.model1(x)
        # x3 = torch.cat([x1,x2],dim=1)
        return x1.squeeze()




class EMESC(nn.Module):
    def __init__(self, Dict_size, hidden_size, cell_size):
        super(EMESC, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.We1 = nn.Sequential(
            # nn.Linear(cell_size,Dict_size),
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wd1 = nn.Sequential(
            # nn.Linear(Dict_size, Dict_size),
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )
        #
        self.We2 = nn.Sequential(
            # nn.Linear(cell_size, Dict_size),
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wd2 = nn.Sequential(
            # nn.Linear(Dict_size, Dict_size),
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wfx1 = nn.Sequential(
            # nn.Linear(Dict_size, Dict_size),
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wix1 = nn.Sequential(
            # nn.Linear(Dict_size, Dict_size),
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wfy1 = nn.Sequential(
            # nn.Linear(cell_size, Dict_size),
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wiy1 = nn.Sequential(
            # nn.Linear(cell_size, Dict_size),
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wfx2 = nn.Sequential(
            # nn.Linear(Dict_size, Dict_size),
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.1)
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),
        )

        self.Wix2 = nn.Sequential(
            # nn.Linear(Dict_size, Dict_size),
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),

            # nn.Linear(Dict_size, Dict_size)
            # nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, stride=1, bias=True),
            # nn.Dropout(0.1)
        )

        self.Wfy2 = nn.Sequential(
            # nn.Linear(cell_size, Dict_size),
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),

            # nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, stride=1, bias=True),
            # nn.Dropout(0.1)
        )

        self.Wiy2 = nn.Sequential(
            # nn.Linear(cell_size, Dict_size),
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            # nn.Dropout(0.5),
            nn.GELU(),
            # nn.Linear(Dict_size, Dict_size),

            # nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, stride=1, bias=True),
            # nn.Dropout(0.1)
        )


        Line1 = [

            nn.Conv2d(in_channels=Dict_size, out_channels=2 * hidden_size, kernel_size=3, stride=1, bias=True),
            #nn.BatchNorm2d(2*hidden_size,eps=1e-8,affine=True,track_running_stats=True),
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Conv2d(in_channels=2 * hidden_size, out_channels=2 * hidden_size, kernel_size=1, stride=1, bias=True),
            #nn.Dropout(0.2),
            #nn.BatchNorm2d(2*hidden_size,eps=1e-8,affine=True,track_running_stats=True),
            # nn.GELU(),
            # nn.Conv2d(in_channels=4*Dict_size, out_channels=2*Dict_size, kernel_size=1, stride=1, bias=True),

            nn.GELU(),
            nn.Conv2d(in_channels=2 * hidden_size, out_channels=1, kernel_size=1, stride=1),
            # nn.ReLU(),

        ]
        self.Line1 = nn.Sequential(*Line1)

        self.activ = nn.Sequential(
            nn.ReLU(),
            # nn.Threshold(0.001, 0, inplace=False),
        )

        self.activ1 = nn.Sequential(
            nn.Sigmoid(),
        )





    def forward(self, y):
        ##init
        ywe1 = self.We1(y)
        ywe2 = self.We2(y)

        ywfy1 = self.Wfy1(y)
        ywiy1 = self.Wiy1(y)

        ywfy2 = self.Wfy2(y)
        ywiy2 = self.Wiy2(y)

        # xt1 = 0
        xt1 = 0
        I = self.activ1(ywiy1)
        c12tlide = -ywe1
        c12 = I * c12tlide
        x12 = self.activ(c12)
        # Spatt = self.Spatialatt(y)
        # Chatt = self.Channelatt(y)

        for i in range(3):
            ## stage2

            xwfx2 = self.Wfx2(x12)
            xwix2 = self.Wix2(x12)

            xwd2 = xt1 - self.Wd2(x12)
            ctlide1 = ywe2 + xwd2  # - 2 + # -x0 2 +x0 +xwd2 2 -xwd2
            ft1 = self.activ1(xwfx2 + ywfy2)
            it1 = self.activ1(xwix2 + ywiy2)

            ct1 = (c12 * ft1 + ctlide1 * it1)
            xt1 = self.activ(ct1)
            ## stage1
            xwfx21 = self.Wfx1(xt1)
            xwix21 = self.Wix1(xt1)

            xwd1 = self.Wd1(xt1)
            c1tlide = ywe1 + xt1 - xwd1  # - 2 + # -x0 2 +x0 +xwd2 2 -xwd2

            ft21 = self.activ1(xwfx21 + ywfy1)
            it21 = self.activ1(xwix21 + ywiy1)
            #
            # ct1 = ct12 * ft1 + cthat1 * it1
            # xt1 = self.activ(ct12)
            #
            ct2 = (ct1 * ft21 + c1tlide * it21)
            xt2 = self.activ(ct2)

            # call[:,:,:,:,i+1]=ct1.clone()
            # xall[:,:,:,:,i+1]=xt1.clone()
            x12 = xt2.clone()
            c12 = ct2.clone()

        xwfx2 = self.Wfx2(x12)
        xwix2 = self.Wix2(x12)
        xwd2 = xt1 - self.Wd2(x12)
        ctlide1 = ywe2 + xwd2  # - 2 +
        ft1 = self.activ1(xwfx2 + ywfy2)
        it1 = self.activ1(xwix2 + ywiy2)
        ct1 = (c12 * ft1 + ctlide1 * it1)

        xt1 = self.activ(ct1)






        x1 = self.Line1(xt1)




        return x1

class EMESC2(nn.Module):
    def __init__(self, Dict_size, hidden_size, cell_size,num_EMESCK):
        super(EMESC2, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size

        # self.EMESCK0 = nn.Sequential(
        #     EMESC(Dict_size,hidden_size,cell_size),
        # )
        #
        # self.EMESCK1 = nn.Sequential(
        #     EMESC(Dict_size,hidden_size,cell_size),
        # )
        # self.EMESCK2 = nn.Sequential(
        #     EMESC(Dict_size,hidden_size,cell_size),
        # )
        # self.EMESCK3 = nn.Sequential(
        #     EMESC(Dict_size,hidden_size,cell_size),
        # )
        self.num_EMESCK = num_EMESCK

        # self.EMESCKs = nn.ModuleList([nn.Sequential(
        #     EMESC(Dict_size, hidden_size, cell_size),
        # ) for i in range(num_EMESCK)])
        self.EMESCKs = nn.ModuleList()
        for i in range(num_EMESCK):
            emesck = nn.Sequential(
                EMESC(Dict_size, hidden_size, cell_size)
            )
            self.EMESCKs.append(emesck)




    def forward(self, y):
        ##init

        # K0 = self.EMESCK0(y)
        # K1 = self.EMESCK1(y)
        # K2 = self.EMESCK2(y)
        # K3 = self.EMESCK3(y)
        # # K4 = self.EMESCK4(y)
        # # K5 = self.EMESCK5(y)
        #
        # K = torch.cat([K0,K1,K2,K3],dim=1)
        # K = K.squeeze()

        Ks = [self.EMESCKs[i](y) for i in range(self.num_EMESCK)]
        # Ks = []
        # for i in range(self.num_EMESCK):
        #     Ks.append(self.EMESCKs[i](y))
        K = torch.cat(Ks, dim=1).squeeze()





        # x2 = self.Line2(K)


        return K

class Lambda(nn.Module):
    def __init__(self,func):
        super(Lambda,self).__init__()
        self.func=func
    def forward(self, input):
        return self.func(input)





class Mapping(nn.Module):
    def __init__(self,n_stage_geneator1=6):
        super(Mapping, self).__init__()

        model = [


            nn.Linear(63,128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 256, bias=True),
            nn.ReLU(),

            nn.Linear(256, 3, bias=True),
            # nn.Threshold(0.0001, 0, inplace=True),
            #nn.ReLU(),
            #nn.Threshold(0.0001, 0, inplace=True),
        ]

        self.model = nn.Sequential(*model)

    def forward(self,x):
        # return torch.log(1+torch.exp(self.model(x)))+1e-10
        return self.model(x)
