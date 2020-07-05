import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self,colordim =3):
        super(Net, self).__init__()
        self.convL1_1 = nn.Conv2d(colordim, 64, 3, padding=1)  # input of (n,n,1), output of (n-2,n-2,64)
        self.convL1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.InL1 = nn.InstanceNorm2d(64)

        self.convL2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convL2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.InL2 = nn.InstanceNorm2d(128)

        self.convL3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convL3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convL3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.InL3 = nn.InstanceNorm2d(256)

        self.convL4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convL4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convL4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.InL4 = nn.InstanceNorm2d(512)

#--------------------------------------------------------------
        self.convL6_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convL6_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.InL6 = nn.InstanceNorm2d(512)
#--------------------------------------------------------------

        self.upconvL8 = nn.Conv2d(1024, 512, 1)
        self.convL8 = nn.Conv2d(512, 256, 3, padding=1)
        self.InL8 = nn.InstanceNorm2d(256)

        self.upconvL9 = nn.Conv2d(512, 256, 1)
        self.convL9 = nn.Conv2d(256, 128, 3, padding=1)
        self.InL9 = nn.InstanceNorm2d(128)

        self.upconvL10 = nn.Conv2d(256, 128, 1)
        self.convL10 = nn.Conv2d(128, 64, 3, padding=1)
        self.InL10 = nn.InstanceNorm2d(64)

        self.upconvL11 = nn.Conv2d(128, 64, 1)
        self.InL11 = nn.InstanceNorm2d(64)
        self.upconvL12 = nn.Conv2d(64, 3, 1)
        self.InL12 = nn.InstanceNorm2d(3)  
        self.upconvL13 = nn.Conv2d(6, 3, 1)
        self.InL13 = nn.InstanceNorm2d(3)


#---------------------------------------------------------------
        self.convD1_1 = nn.Conv2d(colordim, 64, 3, padding=1)  # input of (n,n,1), output of (n-2,n-2,64)
        self.convD1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.InD1 = nn.InstanceNorm2d(64)

        self.convD2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.convD2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.InD2 = nn.InstanceNorm2d(128)

        self.convD3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.convD3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.convD3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.InD3 = nn.InstanceNorm2d(256)

        self.convD4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.convD4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.convD4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.InD4 = nn.InstanceNorm2d(512)

#--------------------------------------------------------------
        self.convD6_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.convD6_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.InD6 = nn.InstanceNorm2d(512)
#--------------------------------------------------------------

        self.upconvD8 = nn.Conv2d(1024, 512, 1)
        self.convD8 = nn.Conv2d(512, 256, 3, padding=1)
        self.InD8 = nn.InstanceNorm2d(256)

        self.upconvD9 = nn.Conv2d(512, 256, 1)
        self.convD9 = nn.Conv2d(256, 128, 3, padding=1)
        self.InD9 = nn.InstanceNorm2d(128)

        self.upconvD10 = nn.Conv2d(256, 128, 1)
        self.convD10 = nn.Conv2d(128, 64, 3, padding=1)
        self.InD10 = nn.InstanceNorm2d(64)

        self.upconvD11 = nn.Conv2d(128, 64, 1)
        self.InD11 = nn.InstanceNorm2d(64)

        self.upconvD12 = nn.Conv2d(64, 3, 1)
        self.InD12 = nn.InstanceNorm2d(3)

        self.upconvD13 = nn.Conv2d(6, 3, 1)
        self.InD13 = nn.InstanceNorm2d(3)




#------------------------------------------------------------------------------        
        self.convC = nn.Conv2d(colordim*3,colordim,1)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        # self.avgpool = nn.AvgPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2) 
        

 
    def forward(self, x0, mask_L, mask_D):

#-----------------------------------------------------------------------------------   

        x1_L = F.leaky_relu(self.InL1(self.convL1_2(F.leaky_relu(self.convL1_1(x0))))) # 64, 320 
        x2_L = F.leaky_relu(self.InL2(self.convL2_2(F.leaky_relu(self.convL2_1(self.maxpool(x1_L)))))) # 128, 160
        x3_L = F.leaky_relu(self.InL3(self.convL3_3(F.leaky_relu(self.convL3_2(F.leaky_relu(self.convL3_1(self.maxpool(x2_L)))))))) # 256, 80 
        x4_L = F.leaky_relu(self.InL4(self.convL4_3(F.leaky_relu(self.convL4_2(F.leaky_relu(self.convL4_1(self.maxpool(x3_L))))))))# 512, 40
        

#-----------------------------------------------------------------------------------
        xup_L = F.leaky_relu(self.InL6(self.convL6_2(F.leaky_relu(self.convL6_1(self.maxpool(x4_L))))))  # 512, 10 
   
# #-----------------------------------------------------------------------------------   

        xup_L = F.leaky_relu(self.InL8(self.convL8(self.upconvL8(torch.cat((self.upsample(xup_L), x4_L),1))))) # 256, 40
        xup_L = F.leaky_relu(self.InL9(self.convL9(self.upconvL9(torch.cat((self.upsample(xup_L), x3_L),1)))))# 256, 80
        xup_L = F.leaky_relu(self.InL10(self.convL10(self.upconvL10(torch.cat((self.upsample(xup_L), x2_L),1))))) # 128, 160
        xup_L = F.leaky_relu(self.InL11(self.upconvL11(torch.cat((self.upsample(xup_L), x1_L),1)))) # 64, 320
        xup_L = F.leaky_relu(self.InL12(self.upconvL12(xup_L))) # 3, 320
        xup_L = F.relu(self.upconvL13(torch.cat((xup_L, x0),1))) # 3, 320

# ----------------------------------------------------------------------------------------
#       
        x1_D = F.leaky_relu(self.InD1(self.convD1_2(F.leaky_relu(self.convD1_1(x0)))))# 64, 320 
        x2_D = F.leaky_relu(self.InD2(self.convD2_2(F.leaky_relu(self.convD2_1(self.maxpool(x1_D)))))) # 128, 160
        x3_D = F.leaky_relu(self.InD3(self.convD3_3(F.leaky_relu(self.convD3_2(F.leaky_relu(self.convD3_1(self.maxpool(x2_D))))))))# 256, 80 
        x4_D = F.leaky_relu(self.InD4(self.convD4_3(F.leaky_relu(self.convD4_2(F.leaky_relu(self.convD4_1(self.maxpool(x3_D)))))))) # 512, 40

#-----------------------------------------------------------------------------------
        xup_D = F.leaky_relu(self.InD6(self.convD6_2(F.leaky_relu(self.convD6_1(self.maxpool(x4_D))))))  # 512, 10 
#-----------------------------------------------------------------------------------   
 
        xup_D = F.leaky_relu(self.InD8(self.convD8(self.upconvD8(torch.cat((self.upsample(xup_D), x4_D),1))))) # 256, 40
        xup_D = F.leaky_relu(self.InD9(self.convD9(self.upconvD9(torch.cat((self.upsample(xup_D), x3_D),1))))) # 256, 80
        xup_D = F.leaky_relu(self.InD10(self.convD10(self.upconvD10(torch.cat((self.upsample(xup_D), x2_D),1))))) # 128, 160
        xup_D = F.leaky_relu(self.InD11(self.upconvD11(torch.cat((self.upsample(xup_D), x1_D),1)))) # 64, 320
        xup_D = F.leaky_relu(self.InD12(self.upconvD12(xup_D))) # 3, 320
        xup_D = self.upconvD13(torch.cat((xup_D, x0),1))
#----------------------------------------------------------------------------------- 
        ones = torch.tensor(1)
        if torch.cuda.is_available():
            ones = ones.cuda()
        x_cat = xup_D*mask_D+xup_L*mask_L+x0*(ones-mask_L-mask_D)

        return x_cat, xup_L, xup_D

 



