# -*- coding: utf-8 -*-

#=== Settings =================================================================

import imageio, cv2
import numpy as np

def func(img):
  img = img**0.9
  return (1+0.6)*img/(img+0.6)

def func_inver(img):
  return np.power(np.divide(0.6*img, np.maximum(1.6-img, 1e-10) ), 1.0/0.9)

def tonemap(images, MU = 5000): 
    return np.log(1 + MU * images) / np.log(1 + MU) 

def itonemap(images, MU = 5000): 
    return  ((1 + MU) ** images - 1) / MU

# low_bound, high_bound
# 
def random_crop(img, sub_im_sc = [6,6], resize = False, rez_im_sc = [320, 320], random_flipping = True):
  img = img.astype(np.float32)

  sub_im_sc = np.array(sub_im_sc)
  img_size = np.array(img.shape) # get size
  
  if sum(img_size[:2]<320)>=1:
    print('img size error(too small)!')
    raise IndexError

  crop_size = (2+(sub_im_sc-2)*np.random.rand(2)).astype(np.int)*160 
  # print('crop_size: ', crop_size)
  # print('img_size: ', img_size)
  crop_size[crop_size>=img_size[:2]] = 320
  h,w = crop_size
  # print(h,w)
  h_start, w_start = (np.random.rand(2)*(img_size[:2]-[h,w])).astype(np.int) 
  # print(h_start,w_start)


  img_crop = img[h_start:h_start+h, w_start:w_start+w,:]

  if resize==True: 
    img_crop = cv2.resize(img_crop, (rez_im_sc[0], rez_im_sc[1]))

  if random_flipping==True:
    if np.random.rand()<0.5:
      img_crop = cv2.flip(img_crop, 1)

  return img_crop

 # mean and std
def random_hue_sat(img, im_hue = [0.0, 3.5], im_sat = [0.0, 0.05]):
  # img = img.astype(np.float32)
  gamma = 0.5
  img = func(img)
  # print('--------------------------------')
  # print(np.max(img))
  # print(np.min(img))

  hue = np.random.normal(im_hue[0], im_hue[1])
  sat = np.random.normal(im_sat[0], im_sat[1])
  hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
  hsv[:,:,0] = hsv[:,:,0]+hue
  hsv[:,:,1] = hsv[:,:,1]+sat
  img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
  # print(np.max(img))
  # print(np.min(img))
  img[img<1e-5] = 1e-5
  img[img>1] = 1
  img = func_inver(img)
  
  return img

def random_contract(img, im_contract = [0.85, 1], im_luminance = [-0.1, 0.1]):
  # img = img.astype(np.float32)
  alpha = im_contract[0] + np.random.rand()*(im_contract[1]-im_contract[0])
  beta = im_luminance[0] + np.random.rand()*(im_luminance[1]-im_luminance[0])
  img = img*alpha+beta
  img[img<1e-5] = 1e-5
  # img[img>1] = 1
  return img
 # mean and std
def random_noise(img, im_noise = [0.0, 0.0001]):
  # img = img.astype(np.float32)
  noise = np.random.normal(im_noise[0], im_noise[1], img.shape).astype(np.float32)
  img = img+noise
  img[img<1e-5] = 1e-5
  img[img>1] = 1 # 
  return img  

 # mean and std
def random_crf(img, im_sigm_n = [0.9, 0.1], im_sigm_a = [0.6, 0.1], test = False):

  if test:
    im_sigm_n[1] = 0
    im_sigm_a[1] = 0
  # img = img.astype(np.float32)
  sigmoid_n = min(2.5, np.random.normal(im_sigm_n[0], im_sigm_n[1]))
  sigmoid_a = min(5, np.random.normal(im_sigm_a[0], im_sigm_a[1]))

  img = img**sigmoid_n
  img = (1+sigmoid_a)*img/(img+sigmoid_a)
  img = img*255
  img[img>255] = 255
  img[img<0] = 0
  img = img.astype(np.uint8)
  return img # 

def random_crf_ACES(img, adapted_lum):
  A = 2.51
  B = 0.03
  C = 2.43
  D = 0.59
  E = 0.14
  color *= adapted_lum
  return img*(A*img+B)/(img*(C*img+D)+E)

# -----------------------------------

def random_overexposure(img, maxvalue, im_oe = [0.85, 0.95]):
  img = img.astype(np.float32)
  img_size = np.array(img.shape)
  gamma = 0.5
  img_ori = img
  img = func(img)
  c_l = im_oe[0]+np.random.rand()*(im_oe[1]-im_oe[0]) # 
  ma = np.max(img)
  mi = np.min(img)
  # print(ma,mi)
  N_r = 256

  img_h = img.reshape(1,-1)
  hist, r = np.histogram(img_h, N_r, [mi, ma]) # 
  num = img_size[0]*img_size[1]*img_size[2]
  c = 0
  for k in range(N_r):
    c0 = c
    c = c+hist[k]/num #  
    if c>=c_l:
      k_l = k
      break # 
  percent = max(0, min(1, (c_l-c0)/(c-c0)))
  sc = percent*r[min(k_l+1,N_r)]+(1-percent)*r[k_l]
  sc_infor = str(sc)[:8]
  ma_infor = str(maxvalue)
  # print("sc in ldr domain: ", sc)
  
  sc = func_inver(sc)
  # print(ma,np.mean(img),mi)
  # print(sc**0.5,k_l,c0,c_l,c)
  # print('----------------')

  img = img_ori/sc # ?
  
  # imageio.imwrite("./view_patch/{}_{}_p_ori.hdr".format(sc_infor, ma_infor), img, format = 'hdr')
  # imageio.imwrite("./view_patch/{}_{}_p_ori.png".format(sc_infor, ma_infor), func(img_ori))

  # bitvalue = 255
  # img = np.clip(img, 0, 1)
  # img_tm = func(img)*bitvalue
  # img_tm = img_tm.astype(np.uint8)
  # imageio.imwrite("./view_patch/{}_{}_p.png".format(sc_infor, ma_infor), img_tm)
  # img_tm = img_tm.astype(np.float32)
  # imageio.imwrite("./view_patch/{}_{}_p.hdr".format(sc_infor, ma_infor), func_inver(img_tm/bitvalue), format = 'hdr')
  # img[img>1] = 1

  return img, sc

 
def random_underexposure(img, im_ue = [0.02, 0.05]):
  img = img.astype(np.float32)
  img_size = np.array(img.shape)
  gamma = 0.5
  img_ori = img
  img = func(img)
  c_d = im_ue[0]+np.random.rand()*(im_ue[1]-im_ue[0]) # 
  ma = np.max(img)
  mi = np.min(img)
  N_r = 256
  # r = []
  # for k in range(N_r+1):
  #   r.append(mi+(ma-mi)*float(k)/N_r)
  img_h = img.reshape(1,-1)
  hist, r = np.histogram(img_h, N_r, [mi, ma]) # 
  num = img_size[0]*img_size[1]*img_size[2]
  c = 0
  for k in range(N_r):
    c0 = c
    c = c+hist[k]/num #  
    if c>=c_d:
      k_d = k
      break # 
  percent = max(0, min(1, (c_d-c0)/(c-c0)))
  sc = percent*r[min(k_d+1,N_r)]+(1-percent)*r[k_d]
  sc = func_inver(sc)
  t = func_inver(1/255) # 
  # print('t: ', t)
  # print('sc: ', sc)
  img = img_ori
  img = img*(t/sc)
  return img
