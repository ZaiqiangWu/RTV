import random

import PIL
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2Img,IUV2SDP,IUV2SSDP
from util.cv2_trans_util import get_inverse_trans
import cv2
import json

class UpperBodyGarment(data.Dataset):
    def __init__(self,path,img_size=512):
        self.simplified_dp=True
        self.img_dir = path
        self.image_list = self.__get_image_list()
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.randomaffine = RandomAffineMatrix(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.5), shear=(-5, 5, -5, 5))
        #self.randomaffine = RandomAffineMatrix(degrees=20, translate=(0.1, 0.1), scale=(1.5, 2.0), shear=(-5, 5, -5, 5))
        with open(os.path.join(self.img_dir,'dataset_info.json'), 'r') as f:
            dataset_info=json.load(f)
        self.raw_height=dataset_info['height']
        self.raw_width=dataset_info['width']

    def __getitem__(self, index):
        garment_path = self.image_list[index]
        garment_img = np.array(Image.open(garment_path))
        raw_h,raw_w =self.raw_height, self.raw_width
        trans2roi_path = os.path.join(self.img_dir,
                                      (os.path.basename(self.image_list[index])).split('_')[0] + '_trans2roi.npy')
        trans2roi = np.load(trans2roi_path)
        inv_trans = get_inverse_trans(trans2roi)
        garment_img = cv2.warpAffine(garment_img, inv_trans, (raw_w, raw_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))



        vm_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_vm.jpg')
        vm_img = np.array(Image.open(vm_path))
        vm_img = cv2.resize(vm_img, (raw_w, raw_h))
        mask_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_mask.png')
        mask_img = np.array(Image.open(mask_path))
        mask_img = cv2.resize(mask_img, (raw_w, raw_h))
        iuv_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_iuv.npy')
        IUV =np.load(iuv_path)

        dp_img = IUV2SDP(IUV)
        dp_img = cv2.resize(dp_img, (raw_w, raw_h),cv2.INTER_NEAREST)

        new_trans2roi = self.randomaffine(trans2roi)

        roi_garment_img = cv2.warpAffine(garment_img, new_trans2roi, (1024,1024),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))
        roi_dp_img = cv2.warpAffine(dp_img, new_trans2roi, (1024, 1024),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        roi_vm_img = cv2.warpAffine(vm_img, new_trans2roi, (1024, 1024),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        roi_mask_img = cv2.warpAffine(mask_img, new_trans2roi, (1024, 1024),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0))
        if self.transform is not None:
            roi_garment_img = self.transform(PIL.Image.fromarray(roi_garment_img))
            roi_vm_img = self.transform(PIL.Image.fromarray(roi_vm_img))
            roi_mask_img = self.transform(PIL.Image.fromarray(roi_mask_img))
            roi_dp_img = self.transform(PIL.Image.fromarray(roi_dp_img))
        if torch.cuda.is_available():
            roi_garment_img = roi_garment_img.cuda()
            roi_vm_img = roi_vm_img.cuda()
            roi_mask_img = roi_mask_img.cuda()
            roi_dp_img = roi_dp_img.cuda()
        return self._normalize(roi_garment_img),self._normalize(roi_vm_img),self._normalize(roi_dp_img),roi_mask_img

    def _normalize(self, x):
        # map from 0,1 to -1,1
        return x * 2.0 - 1.0


    def __get_image_list(self):
        image_list = []
        filelist = os.listdir(self.img_dir)
        for item in filelist:
            if item.endswith('_garment.jpg'):
                image_list.append(os.path.join(self.img_dir, item))
        image_list.sort()
        return image_list

    def __len__(self):
        return len(self.image_list)

class RandomAffineMatrix:
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, trans:np.ndarray):
        R_hat,t_hat=get_random_affine_params(self.degrees, self.translate, self.scale, self.shear)
        new_trans=self.deform(R_hat,t_hat,trans)
        return new_trans

    def batch_forward(self, trans_list):
        R_hat,t_hat=get_random_affine_params(self.degrees, self.translate, self.scale, self.shear)
        new_list =[]
        for trans in trans_list:
            new_trans=self.deform(R_hat,t_hat,trans)
            new_list.append(new_trans)
        return new_list

    def deform(self,R_hat,t_hat,trans):
        c = np.array([512, 512])
        R = trans[:, :2]
        t = trans[:, 2]
        R_new = np.dot(R_hat, R)
        t_new = np.dot(R_hat, t - c) + t_hat + c
        new_trans = np.concatenate((R_new, t_new[:, None]), axis=1)
        return new_trans



def get_random_affine_params(degrees, translate=None, scale=None, shear=None):
    # Random rotation angle
    angle = np.random.uniform(-degrees, degrees)
    angle_rad = np.deg2rad(angle)
    img_size=1024

    # Random translation
    if translate is not None:
        max_dx = translate[0]*img_size
        max_dy = translate[1]*img_size
        tx = np.random.uniform(-max_dx, max_dx)
        ty = np.random.uniform(-max_dy, max_dy)
    else:
        tx, ty = 0, 0

    # Random scaling
    if scale is not None:
        scale_factor = np.random.uniform(scale[0], scale[1])
    else:
        scale_factor = 1.0

    # Random shear
    if shear is not None:
        shear_x = np.random.uniform(shear[0], shear[1])
        shear_y = np.random.uniform(shear[2], shear[3]) if len(shear) > 2 else 0
    else:
        shear_x, shear_y = 0, 0

    # Compute the affine transformation matrix
    cos_theta = np.cos(angle_rad) * scale_factor
    sin_theta = np.sin(angle_rad) * scale_factor
    shear_x_rad = np.deg2rad(shear_x)
    shear_y_rad = np.deg2rad(shear_y)

    # Create the affine transformation matrix
    M = np.array([
        [cos_theta + np.tan(shear_y_rad) * sin_theta, -sin_theta + np.tan(shear_y_rad) * cos_theta],
        [sin_theta + np.tan(shear_x_rad) * cos_theta, cos_theta + np.tan(shear_x_rad) * sin_theta]
    ])

    # Translation vector
    t = np.array([tx, ty])

    return M, t

