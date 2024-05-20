import torch
import torch.nn as nn


class tEncoder(nn.Module):
    def __init__(self,feat):
        super(tEncoder, self).__init__()
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


        y1 = self.activ0(x2)

        return y1


class Mappingt(nn.Module):
    def __init__(self):
        super(Mappingt, self).__init__()
        input=256
        model = [
            nn.Linear(input, 2 * input, bias=True),

            nn.GELU(),
            nn.Linear(2 * input, 4 * input, bias=True),
            nn.GELU(),

            nn.Linear(4 * input, 2 * input, bias=True),
            nn.GELU(),
            nn.Linear(2 * input, 2 , bias=True),


            nn.ReLU(),
        ]

        self.model = nn.Sequential(*model)



    def forward(self,x):
        x1 = self.model(x)

        return x1.squeeze()




class qEncoder(nn.Module):
    def __init__(self, Dict_size, hidden_size, cell_size):
        super(qEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.We1 = nn.Sequential(
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            nn.GELU(),
        )

        self.Wd1 = nn.Sequential(
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            nn.GELU(),
        )
        #
        self.We2 = nn.Sequential(
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            nn.GELU(),
        )

        self.Wd2 = nn.Sequential(
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            nn.GELU(),
        )

        self.Wfx1 = nn.Sequential(
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            nn.GELU(),
        )

        self.Wix1 = nn.Sequential(
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            nn.GELU(),
        )

        self.Wfy1 = nn.Sequential(
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            nn.GELU(),
        )

        self.Wiy1 = nn.Sequential(
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            nn.GELU(),
        )

        self.Wfx2 = nn.Sequential(
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            nn.GELU(),
        )

        self.Wix2 = nn.Sequential(
            nn.Conv2d(in_channels=Dict_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),
            nn.GELU(),

        )

        self.Wfy2 = nn.Sequential(
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            nn.GELU(),

        )

        self.Wiy2 = nn.Sequential(
            nn.Conv2d(in_channels=cell_size, out_channels=Dict_size, kernel_size=1, padding=0, padding_mode='reflect',
                      stride=1, bias=True),

            nn.GELU(),

        )


        Line1 = [

            nn.Conv2d(in_channels=Dict_size, out_channels=2 * hidden_size, kernel_size=3, stride=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=2 * hidden_size, out_channels=2 * hidden_size, kernel_size=1, stride=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=2 * hidden_size, out_channels=1, kernel_size=1, stride=1),

        ]
        self.Line1 = nn.Sequential(*Line1)

        self.activ = nn.Sequential(
            nn.ReLU(),
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

        xt1 = 0
        I = self.activ1(ywiy1)
        c12tlide = -ywe1
        c12 = I * c12tlide
        x12 = self.activ(c12)

        for i in range(3):

            xwfx2 = self.Wfx2(x12)
            xwix2 = self.Wix2(x12)

            xwd2 = xt1 - self.Wd2(x12)
            ctlide1 = ywe2 + xwd2
            ft1 = self.activ1(xwfx2 + ywfy2)
            it1 = self.activ1(xwix2 + ywiy2)

            ct1 = (c12 * ft1 + ctlide1 * it1)
            xt1 = self.activ(ct1)
            ## stage1
            xwfx21 = self.Wfx1(xt1)
            xwix21 = self.Wix1(xt1)

            xwd1 = self.Wd1(xt1)
            c1tlide = ywe1 + xt1 - xwd1

            ft21 = self.activ1(xwfx21 + ywfy1)
            it21 = self.activ1(xwix21 + ywiy1)

            #
            ct2 = (ct1 * ft21 + c1tlide * it21)
            xt2 = self.activ(ct2)

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

class qEncoders(nn.Module):
    def __init__(self, Dict_size, hidden_size, cell_size,num_qEncoder):
        super(qEncoders, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.num_qEncoder = num_qEncoder
        self.qEncoders = nn.ModuleList()
        for i in range(num_qEncoder):
            qencoder = nn.Sequential(
                qEncoder(Dict_size, hidden_size, cell_size)
            )
            self.qEncoders.append(qencoder)


    def forward(self, y):

        Ks = [self.qEncoders[i](y) for i in range(self.num_qEncoder)]
        K = torch.cat(Ks, dim=1).squeeze()
        return K


