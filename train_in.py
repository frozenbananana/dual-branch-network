from __future__ import division
import torch, json, pdb, cv2, os, imageio, time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import sys

import torch.nn.functional as F
from get_ldri import *
from network_sp_in import *
from torch.utils.data import Dataset, DataLoader

esp = 1e-5
DIR = '/data/HDRwork/hdr_data/'
OUT_DIR = sys.argv[1] # './out_catnet'
# w_loss = [float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])] # [w_L, w_D, w_M]
save_name = OUT_DIR+'_para.pkl'
HDR_PATH = '/data/HDRwork/hdr_data/train_gt' # os.path.join(DIR, 'train_h', 'hdr')
LDR_PATH = os.path.join(DIR, 'train_h', 'jpg')
HDR_PATH_VAL = os.path.join(DIR, 'val_h', 'hdr')
LDR_PATH_VAL = os.path.join(DIR, 'val_h', 'jpg')

model_name = 'v7_hue_para.pkl'  
size_ori = [320, 320]
ta = 0.88
tb = 0.12 # 0.1
w_scale = 1
gamma = 2

#super parameter
LR=0.00001 # 0.00005  
BATCH_SIZE=20
EPOCHES=5000

class HDRDataset(Dataset):
 
    def __init__(self, hdr_path, ldr_path, aug_online = False):

        self.hdr_path = hdr_path
        self.ldr_path = ldr_path
        self.img_list = os.listdir(hdr_path)
        self.gamma = 2
        self.aug_online = aug_online
        self.darkness = 0.08
    def __len__(self):

        return len(self.img_list)

    def __getitem__(self, index):
        name = self.img_list[index][:-4]
        
        img_y = imageio.imread(os.path.join(self.hdr_path, self.img_list[index]))
     
        img_y = img_y.astype(np.float32)
        if self.aug_online:
            try:
                [img_y, img_x] = self.augment_online(img_y)
            except:
                print(name)
                raise("something wrong!!")
        else:
            img_x = imageio.imread(os.path.join(self.ldr_path, name+'.jpg')) # 
        img_x = img_x.astype(np.float32)
        imageio.imwrite(os.path.join("mask_save", name+".png"), img_x)

        img_x = img_x/255
        luminance = 0.299*img_x[:,:,0]+0.587*img_x[:,:,1]+0.114*img_x[:,:,2] # (np.max(img_x, 2)+np.min(img_x, 2))/2
        img_mask_L = np.minimum(np.maximum(0, luminance-ta)/(1-ta/w_scale),1)
        img_mask_D = np.minimum(np.maximum(0, tb-luminance)/(w_scale*tb),1)
        mask_dark = (1-np.abs(self.darkness-luminance)/max(tb-self.darkness, self.darkness))
        mask_dark = np.clip(mask_dark, 0, 1)
        # img_mask_D *= mask_dark
        
        img_mask_D = (img_mask_D*255).astype(np.uint8)
        mask_nopro = img_mask_D.copy()
        mask_nopro = 255-cv2.threshold(mask_nopro, 25, 255, cv2.THRESH_BINARY)[1]
        mask_gf = cv2.ximgproc.guidedFilter(guide = img_mask_D, src = img_mask_D, radius=16, eps=4000, dDepth=-1)
        mask_nopro = mask_nopro/255
        img_mask_D = img_mask_D*mask_nopro+mask_gf*(1-mask_nopro)
        img_mask_D = img_mask_D.astype(np.float32)
        # cv2.imwrite(os.path.join("mask_save", name+"mask_L.png"), img_mask_L*255)
        # cv2.imwrite(os.path.join("mask_save", name+"mask_D.png"), img_mask_D)
        img_mask_D = img_mask_D/255

        
        img_mask_L = img_mask_L[:,:,np.newaxis]
        img_mask_D = img_mask_D[:,:,np.newaxis]
        img_x = torch.Tensor(img_x)
        img_x = self.func_inver(img_x)
        img_x = img_x.permute(2,0,1)
        img_y = torch.Tensor(img_y).permute(2,0,1)
        img_mask_L = torch.Tensor(img_mask_L).permute(2,0,1)
        img_mask_D = torch.Tensor(img_mask_D).permute(2,0,1)

        return {'img_ldr':img_x, 'img_hdr': img_y, 'img_mask_L':img_mask_L, 'img_mask_D':img_mask_D, 'name':name}


    def augment_online(self, img):

        resize = True
        r_hue = 0.5
        r_noise = 0
        r_h = 1
        im_oe = [0.95,0.98] #[0.95,0.98] 
        im_ue = [0.02, 0.05]
        im_contract = [0.85, 95]
        im_luminance = [0, 0]
        img_size = np.array(img.shape)

        if sum(img_size<320)>=2:
            raise Exception('img size is too small!')
        
        img_hdr = random_crop(img, resize = resize)
        maxvalue = np.max(img_hdr)
        img_hdr = img_hdr/maxvalue #    
        if np.random.rand()<r_hue:
            img_hdr = random_hue_sat(img_hdr)
        if np.random.rand()<r_noise:
            img_hdr = random_noise(img_hdr)
        if np.random.rand()<r_h:
            img_hdr, sc = random_overexposure(img_hdr, maxvalue, im_oe = im_oe)
        # else:
        #     img_hdr = random_contract(img_hdr, im_contract = im_contract, im_luminance = im_luminance)
            # img_hdr[img_hdr>1] = 1 
       
            # img_hdr = random_underexposure(img_hdr, im_ue = [0.02, 0.05])
        img_hdr_y = img_hdr
        img_hdr = np.clip(img_hdr, 0, 1)

        return [img_hdr_y, random_crf(img_hdr)]


    def func_inver(self, img):
        return torch.pow(torch.div(0.6*img, torch.max(1.6-img, torch.tensor(1e-10))), 1.0/0.9)



def trainOneBatch(batch:torch.FloatTensor,raw:torch.FloatTensor,mask_L:torch.FloatTensor,mask_D:torch.FloatTensor):
    x_cat, xup_L, xup_D =auto(batch,mask_L,mask_D)
    loss_L,loss_color_L = loss_function_L(xup_L, raw, mask_L)
    loss_D,loss_color_D = loss_function_D(xup_D, raw, mask_D)
    # loss_D = loss_D/tb
    # loss_C = w_loss[0]*loss_function_L(x_cat, raw, mask_L)+w_loss[1]*loss_function_D(x_cat, raw, mask_D)/tb

    # loss = loss_L + loss_D 
     
    # train dark branch only-----------
    loss =  loss_D
    # ---------------------------------
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), loss_L.item(), loss_D.item(), loss_color_L.item(), loss_color_D.item(), x_cat
#
def testOneBatch(batch:torch.FloatTensor,raw:torch.FloatTensor,mask_L:torch.FloatTensor,mask_D:torch.FloatTensor):
    x_cat, xup_L, xup_D =auto(batch,mask_L,mask_D)
    loss_L,loss_color_L = loss_function_L(xup_L, raw, mask_L)
    loss_D,loss_color_D = loss_function_D(xup_D, raw, mask_D)
    # loss_D = loss_D/tb
    loss = loss_L + loss_D
    return loss.item(), loss_L.item(), loss_D.item() , loss_color_L.item(), loss_color_D.item(), x_cat
 
 

def loss_function_L(y_pre, y_gt, mask, lamda = 0.01, cuda_available = True):
    # mseloss = nn.MSELoss()
    # mask = mask**0.5
    loss_mse = (torch.log(y_pre+esp) - torch.log(y_gt+esp))**2 * mask
    sum_value = (mask>0).sum()
    if sum_value==0:
        sum_value=1
    loss_mse = loss_mse.sum()/sum_value
    # loss_mse = loss_mse.mean()

    # loss_color = 0
    loss_color = (get_hue_value(y_pre) - get_hue_value(y_gt))**2 * mask
    loss_color = loss_color.sum()/sum_value

    loss = loss_mse + lamda*loss_color
    return loss,loss_color
def loss_function_D(y_pre, y_gt, mask, lamda = 0.01, cuda_available = True):
    # mseloss = nn.MSELoss()

    # mask = mask**0.5
    # loss_mse = mseloss(y_pre*mask/tb, y_gt*mask/tb)
    loss_mse = (y_pre-y_gt)**2*mask
    sum_value = (mask>0).sum()
    if sum_value==0:
        sum_value=1
    loss_mse = loss_mse.sum()/sum_value
    # loss_mse = loss_mse.mean()
    loss_color = (get_hue_value(y_pre) - get_hue_value(y_gt))**2*mask
    loss_color = loss_color.sum()/sum_value
    loss = loss_mse
    return loss,loss_color

def loss_function_M(y_pre, y_gt, mask, lamda = 0.001, cuda_available = True):
    mseloss = nn.MSELoss()
    mask = mask**0.5
    loss_mse = mseloss(y_pre*mask, y_gt*mask)
    loss_color = mseloss(get_hue_value(y_pre)*mask, get_hue_value(y_gt)*mask)
    loss = lamda*loss_color+loss_mse
    return loss

def get_hue_value(img):
    H = img
    

    img_max, index_max = torch.max(img, 1)
    img_min, index_min = torch.min(img, 1)

    temp = torch.zeros(img_max.shape).cuda()
    img_r = img[:,0,:,:]
    img_g = img[:,1,:,:]
    img_b = img[:,2,:,:]
    d = img_max-img_min
    index_nz = d!=0
    img_g_sel = img_g[index_nz]
    img_b_sel = img_b[index_nz]
    d_sel = d[index_nz]

    temp[index_nz] = 60*(img_g_sel-img_b_sel )/d_sel # 
    img_hue = temp
    # index = (index_max==0)*(img_g<img_b) # 
    # img_hue[index] = temp[index]+360
    index2 = (index_max==1)*index_nz
    img_b_sel2 = img_b[index2]
    img_r_sel2 = img_r[index2]
    d_sel2 = d[index2]
    img_hue[index2] = 60*(img_b_sel2-img_r_sel2)/d_sel2+120

    index3 = (index_max==2)*index_nz
    img_r_sel3 = img_r[index3]
    img_g_sel3 = img_g[index3]
    d_sel3 = d[index3]

    img_hue[index3] = 60*(img_r_sel3-img_g_sel3)/d_sel3+240 
    zeros = torch.zeros_like(img_hue).cuda()
    zeros[img_hue<0] = 360
    img_hue = img_hue+zeros
    # print(img_hue.shape)
    return img_hue/360

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

val_files = os.listdir(OUT_DIR)
if len(val_files)!=0:
    print('need to delete files...')
    for file in val_files:
        os.remove(os.path.join(OUT_DIR, file))



cuda_available=torch.cuda.is_available()
if cuda_available==True:
    print('gpu is available!')

auto = Net() # AutoEncoder() 
if model_name!='':
    print('load pretrained model....')
    state_dict = torch.load(model_name, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    # load params
    model_dict = auto.state_dict()
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
    model_dict.update(new_state_dict)
    auto.load_state_dict(model_dict)

    print("params frozen...")
    for name, para in auto.named_parameters():
        print(name)
        if 'L' in name:
            print('cut!')
            para.requires_grad = False


    print('done!')
if cuda_available :
    auto = torch.nn.DataParallel(auto)
    auto.cuda()
 
#
optimizer=torch.optim.Adam(auto.parameters(),lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

VAL_LOSS = [100] # 
best_epoch = 0
print('getting train data...')
data = HDRDataset(HDR_PATH, LDR_PATH, aug_online = True)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
print('getting val data...')
val_data = HDRDataset(HDR_PATH_VAL, LDR_PATH_VAL)
val_dataloader = DataLoader(val_data, num_workers=4)
print('training begin...')
#
k = -1
print_test_loss = 0
while k<=EPOCHES:
    k = k+1
    print('%d/%d'%(k,EPOCHES))

    count = 0 # 
    batch = []
    raw = []
    mask_light = []
    print_loss = 0
    for i, sample in enumerate(dataloader):
        raw_train = sample['img_hdr']
        batch_train = sample['img_ldr']
        img_mask_D = sample['img_mask_D']
        img_mask_L = sample['img_mask_L']


        print_loss = print_loss + 1
    
        if cuda_available:
            raw_train=raw_train.cuda()#
            batch_train=batch_train.cuda()
            img_mask_D=img_mask_D.cuda()
            img_mask_L=img_mask_L.cuda()
            # print(batch_train.shape)
        loss, loss_L, loss_D, loss_color_L, loss_color_D,decoded = trainOneBatch(batch_train, raw_train, img_mask_L, img_mask_D)#训练一个批次

        if print_loss==50:
            print_loss = 0
            print_test_loss = print_test_loss+1
            print('{}/{} train_loss:{:.8f}, loss_L:{:.4f}, loss_color_L:{:.4f}, loss_D:{:.8f}, loss_color_D:{:.4f}, lr:{:.8f}'.format(k,EPOCHES,loss,loss_L,loss_color_L,loss_D,loss_color_D, scheduler.get_lr()[0]))

            
            if print_test_loss==10:
                print_test_loss = 0
                val_loss = 0
                val_loss_L = 0
                val_loss_D = 0
                val_loss_color_L = 0
                val_loss_color_D = 0
                y_final_all = []
                for i_val, sample_val in enumerate(val_dataloader):
                    val_x = sample_val['img_ldr']
                    val_y = sample_val['img_hdr']
                    val_mask_L = sample_val['img_mask_L']
                    val_mask_D = sample_val['img_mask_D']
                    val_name = sample_val['name'][0]
                    if cuda_available:
                        loss, loss_L, loss_D, loss_color_L, loss_color_D,decoded = testOneBatch(val_x.cuda(), val_y.cuda(), val_mask_L.cuda(), val_mask_D.cuda())
                    else:
                        loss, loss_L, loss_D, loss_color_L, loss_color_D,decoded = testOneBatch(val_x, val_y, val_mask_L, val_mask_D)            
                    if cuda_available:
                        decoded = decoded.cpu().detach().permute(0,2,3,1).numpy().squeeze() # 从gpu上拉下来
                    else:
                        decoded = decoded.detach().permute(0,2,3,1).numpy().squeeze()
                    val_loss = val_loss + loss
                    val_loss_L = val_loss_L + loss_L
                    val_loss_D = val_loss_D + loss_D
                    val_loss_color_L = val_loss_color_L + loss_color_L
                    val_loss_color_D = val_loss_color_D + loss_color_D
            
                    # ------------------------------------------------------------------------------------
                    val_img_x = val_x.permute(0,2,3,1).numpy().squeeze()
        
                    y_final = decoded

                    # #---------------------------------------------------------------------------------------------#
                    y_final = y_final.astype(np.float32)
         

                    y_final_all.append([os.path.join(OUT_DIR, str(k//10)+'_'+val_name+'_'+'y_pre.hdr'), y_final])
                    
                    # ------------------------------------------------------------------------------------
                val_loss = val_loss/(i_val+1)
                val_loss_L = val_loss_L/(i_val+1)
                val_loss_D = val_loss_D/(i_val+1)
                val_loss_color_L = val_loss_color_L/(i_val+1)
                val_loss_color_D = val_loss_color_D/(i_val+1)

                if val_loss<min(VAL_LOSS):
                    print('getting better result!')
                    
                    for val_infor in y_final_all:
                        try:
                            imageio.imwrite(val_infor[0], val_infor[1], format='hdr')
                        except:
                            print('img save failed!')
                            continue

                    best_epoch = k
                    torch.save(auto.state_dict(), save_name) # 

                VAL_LOSS.append(val_loss)
                print('{}/{} test_loss:{:.8f}, loss_L:{:.4f}, loss_color_L:{:.4f}, loss_D:{:.8f}, loss_color_D:{:.4f}, lr:{:.8f}'.format(k,EPOCHES,val_loss,val_loss_L,val_loss_color_L,val_loss_D,val_loss_color_D, scheduler.get_lr()[0]))
                print('%d/%d min_loss: '%(best_epoch,EPOCHES), min(VAL_LOSS))

    if k%50==0 and k!=0:
        scheduler.step()
                # if k%200==0:
                #     for val_infor in y_final_all:
                #         try:
                #             imageio.imwrite(val_infor[0], val_infor[1], format='hdr')
                #         except:
                #             print('img save failed!')
                #             continue
                #     torch.save(auto.state_dict(), str(k)+'_'+save_name)

           


