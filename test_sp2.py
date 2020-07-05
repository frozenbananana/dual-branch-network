from __future__ import division
import torch, json, pdb, cv2, os, imageio, time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import sys
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt
import torch.nn.functional as F
from get_ldri import *
from network_sp_in import *
from torch.utils.data import Dataset, DataLoader
from thop import profile


esp = 1e-5
model_name = sys.argv[1] # 'LightNet_pretrain.pkl', 'LightNet_params_val.pkl'
DIR = sys.argv[2]
HDR_DIR = sys.argv[3]
if HDR_DIR!='0':
    print("HDR input !")
    OUT_DIR = model_name.split('.')[0] + "_" + HDR_DIR.split('/')[-1] + "_out"
else:
    OUT_DIR = model_name.split('.')[0] + "_" + DIR.split('/')[-1] + "_cat_out"

# rlaunch --cpu=8 --memory=50000 python3.5 test.py model_111_min.pkl test_tmp 0 out_test_tmp
 

ta = 0.85
tb = 0.12
gamma = 2
w_scale = 0.95


class LDRDataset(Dataset):
  
    def __init__(self, ldr_path, hdr_path = '0', test=False):

        self.ldr_path = ldr_path
        self.hdr_path = hdr_path
        self.crop = True
        if self.hdr_path!='0':
            self.img_list = os.listdir(hdr_path)
        else:
            self.img_list = os.listdir(ldr_path)
        self.gamma = 2
        self.darkness = 0.08
        self.test = test

    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, index):
        name = self.img_list[index][:-4]

        img_ori = 0
        if self.hdr_path!='0':
            img_x = imageio.imread(os.path.join(self.hdr_path, self.img_list[index]))
            img_x = img_x.astype(np.float32)
            img_x = img_x/img_x.max()
            # if self.crop:
            #     img_x = random_crop(img_x, resize = False)
        
            img_ori = img_x
            img_x, _= random_overexposure(img_x, 1,im_oe = [0.95, 0.95])
            img_x = random_crf(img_x, test = True)/255 # tonemapping??
        else:
            img_x = imageio.imread(os.path.join(self.ldr_path, self.img_list[index])) # 根据list从nori中读数据
            MAXVALUE = np.max(img_x)
            if MAXVALUE>255:
                MAXVALUE = 65535
                print('MAXVALUE: ', MAXVALUE)
            SR = False
            if SR:
                img_x = custom_blur_demo(img_x)
            img_x = img_x.astype(np.float32)
            img_x = img_x/MAXVALUE
  
        w,h,c = img_x.shape

        if w > 1920:
            h = int(h*1920/w)
            w = 1920
        if h > 1920:
            w = int(w*1920/h)
            h = 1920

        w_new = round(w/320)*320
        h_new = round(h/320)*320
        
        if w<320:
            w_new = 320 
        if h<320:
            h_new = 320 

        
        try:
            img_x = cv2.resize(img_x, (h_new, w_new))
            if self.hdr_path!='0':
                img_ori = cv2.resize(img_ori, (h_new, w_new))
            # print('resize -> :' , [h_new, w_new])
        except:
            print(h_new, w_new)

        luminance = 0.299*img_x[:,:,0]+0.587*img_x[:,:,1]+0.114*img_x[:,:,2] # (np.max(img_x, 2)+np.min(img_x, 2))/2

        img_mask_L = np.maximum(0, luminance-ta)/(1-ta)
        img_mask_D = np.maximum(0, tb-luminance)/tb

        mask_dark = (1-np.abs(self.darkness-luminance)/max(tb-self.darkness, self.darkness))
        mask_dark = np.clip(mask_dark, 0, 1)
        # img_mask_D = mask_dark * img_mask_D

        img_mask_D = (img_mask_D*255).astype(np.uint8)
        mask_nopro = img_mask_D.copy()
        mask_nopro = 255-cv2.threshold(mask_nopro, 25, 255, cv2.THRESH_BINARY)[1]
        mask_gf = cv2.ximgproc.guidedFilter(guide = img_mask_D, src = img_mask_D, radius=16, eps=4000, dDepth=-1)
        mask_nopro = mask_nopro/255
        img_mask_D = img_mask_D*mask_nopro+mask_gf*(1-mask_nopro)
        img_mask_D = img_mask_D/255
 

        img_mask_L = img_mask_L[:,:,np.newaxis]
        img_mask_D = img_mask_D[:,:,np.newaxis]
        img_x = torch.Tensor(img_x)
        img_x = self.func_inver(img_x)
        img_x = img_x.permute(2,0,1)
        img_mask_L = torch.Tensor(img_mask_L).permute(2,0,1)
        img_mask_D = torch.Tensor(img_mask_D).permute(2,0,1)

        return {'img_ldr':img_x, 'img_mask_L':img_mask_L, 'img_mask_D':img_mask_D, 'name':name, 'sz_ori':[h,w], 'img_hdr':img_ori}

    def func_inver(self, img):
        # return torch.pow(torch.div(0.6*img, torch.max(1.6-img, torch.tensor(1e00))), 1.0/0.9)
        return torch.pow(img, 2.2)



if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

val_files = os.listdir(OUT_DIR)
if len(val_files)!=0:
    print('need to delete files...')
    for file in val_files:
        os.remove(os.path.join(OUT_DIR, file))


#
cuda_available = torch.cuda.is_available()
if cuda_available==True:
    print('gpu is available!')
#
auto = Net() # AutoEncoder() 
if model_name!='':
    print('load pretrained model....')
    state_dict = torch.load(model_name, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    model_dict = auto.state_dict()
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
    model_dict.update(new_state_dict)
    auto.load_state_dict(model_dict) 
    print('done!')
else:
    raise Exeption('input model~!')
if cuda_available :
    auto = torch.nn.DataParallel(auto)
    auto.cuda()

# w = 1920
# h = 1080
# inputs1 = torch.randn(1, 3, w, h)
# inputs2 = torch.randn(1, 1, w, h)
# inputs3 = torch.randn(1, 1, w, h)
# flops, params = profile(auto, inputs=(inputs1, inputs2, inputs3))

# print("flops: ", flops)
# print("params: ", params)


print('getting test data...')
val_data = LDRDataset(DIR, HDR_DIR, test=True)
val_dataloader = DataLoader(val_data, num_workers=4)


print_test_loss = 0
val_loss = 0
val_loss_L = 0
val_loss_D = 0
val_loss_M = 0
need_time = 0
y_final_all = []
L2_error = []
for i_val, sample_val in enumerate(val_dataloader):
    test_x = sample_val['img_ldr']
    test_name = sample_val['name'][0]
    sz_ori = sample_val['sz_ori']
    img_mask_D = sample_val['img_mask_D']
    img_mask_L = sample_val['img_mask_L']
    img_ori = sample_val['img_hdr']
    if cuda_available:
        test_x = test_x.cuda()
        img_mask_D= img_mask_D.cuda()
        img_mask_L = img_mask_L.cuda()
    print('process..', test_name)

    st = time.time()
    # print(test_x.shape, img_mask_L.shape, img_mask_D.shape)
    x_cat, xup_L, xup_D = auto(test_x, img_mask_L, img_mask_D)

    et = time.time()
    need_time += et-st
    if cuda_available:
        x_cat = x_cat.cpu().detach().permute(0,2,3,1).numpy().squeeze() # 从gpu上拉下来
        test_x = test_x.cpu().detach().permute(0,2,3,1).numpy().squeeze()
        img_mask_D = img_mask_D.cpu().detach().permute(0,2,3,1).numpy().squeeze() 
        img_mask_L = img_mask_L.cpu().detach().permute(0,2,3,1).numpy().squeeze()
        xup_L = xup_L.cpu().detach().permute(0,2,3,1).numpy().squeeze() 
        xup_D = xup_D.cpu().detach().permute(0,2,3,1).numpy().squeeze() 
    else:
        x_cat = x_cat.detach().permute(0,2,3,1).numpy().squeeze()
        test_x = test_x.permute(0,2,3,1).numpy().squeeze()
        img_mask_D = img_mask_D.detach().permute(0,2,3,1).numpy().squeeze() 
        img_mask_L = img_mask_L.detach().permute(0,2,3,1).numpy().squeeze() 
        xup_L = xup_L.detach().permute(0,2,3,1).numpy().squeeze() 
        xup_D = xup_D.detach().permute(0,2,3,1).numpy().squeeze() 
    # ---------------------------------------------保存验证图片---------------------------------------
    y_final = x_cat
    # #---------------------------------------------------------------------------------------------#
    # test_x = test_x.permute(0,2,3,1).numpy().squeeze()
    # test_x = func(test_x)*255
    # test_x = test_x.astype(np.uint8)
    y_final = y_final.astype(np.float32)
    img_ori = img_ori.numpy()

    err = (y_final-img_ori)**2
    L2_error.append(err.mean())

 
    img_mask_D = img_mask_D[:, :, np.newaxis]
    xup_D = xup_D*img_mask_D+(1-img_mask_D)*test_x

    img_mask_L = img_mask_L[:, :, np.newaxis]
    xup_L = xup_L*img_mask_L+(1-img_mask_L)*test_x

    infor = [os.path.join(OUT_DIR, test_name), y_final]
    print('size: ', sz_ori)
    test_x = cv2.resize(test_x,(sz_ori[0], sz_ori[1]))
    y_final = cv2.resize(y_final,(sz_ori[0], sz_ori[1]))
    xup_L = cv2.resize(xup_L,(sz_ori[0], sz_ori[1]))
    # img_mask_L = cv2.resize(img_mask_L,(sz_ori[0], sz_ori[1]))
    xup_D = cv2.resize(xup_D,(sz_ori[0], sz_ori[1]))
    img_mask_D = cv2.resize(img_mask_D,(sz_ori[0], sz_ori[1]))

    tonemapDurand = cv2.createTonemapDurand(2)
    y_final_ldr = tonemapDurand.process(y_final)
    y_final_ldr = np.clip(y_final_ldr, 0, 1)

    try: 
        # xup_L = xup_L**0.8
        imageio.imwrite(infor[0]+'.hdr', y_final, format='hdr')
        # imageio.imwrite(infor[0]+'.jpg', y_final_ldr*255)
        # imageio.imwrite(infor[0]+'mask_L.jpg', img_mask_L*255)
        # imageio.imwrite(infor[0]+'_x.hdr', test_x, format='hdr')
        # imageio.imwrite(infor[0]+'_xup_D.hdr', xup_D, format='hdr')
    except:
        print('img save failed!')
        continue
print(L2_error)
print(np.mean(L2_error))
print("need_time: ", need_time/len(L2_error))    
    # ------------------------------------------------------------------------------------


         
