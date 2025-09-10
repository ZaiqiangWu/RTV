import os
import threading

import numpy as np
import cv2

import util.util
from options.test_options import TestOptions
from model.pix2pixHD.models import create_model
import util.util as util
import torch
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from SMPL.smpl_regressor import SMPL_Regressor
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from util.image_warp import zoom_in, zoom_out, shift_image_right, shift_image_down, rotate_image
from util.image_process import blur_image
#from composition.short_sleeve_composition import ShortSleeveComposer
from model.DensePose.densepose_extractor import DensePoseExtractor
import time

from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from tqdm import tqdm
from composition.naive_overlay import naive_overlay, naive_overlay_alpha
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg, IUV2SDP
from threading import Thread


def make_pix2pix_model(name, input_nc, output_nc=3, model_name='pix2pixHD',ckpt_dir=None):
    opt = TestOptions().parse(save=False, use_default=True, show_info=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.name = name
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    opt.isTrain = False
    opt.model = model_name
    if ckpt_dir is None:
        opt.checkpoints_dir='./rtv_ckpts'
    else:
        opt.checkpoints_dir=ckpt_dir
    opt.gpu_ids=''#load to cpu
    model = create_model(opt)
    # print(model)
    return model


class FrameProcessor:
    def __init__(self, garment_name_list,ckpt_dir=None):
        self.smpl_regressor = SMPL_Regressor(use_bev=True)
        self.viton_model = None
        self.ckpt_dir = ckpt_dir
        self.densepose_extractor = DensePoseExtractor()
        self.upper_body = UpperBodySMPL()
        self.garment_name_list = garment_name_list#[2, 3, 17, 18, 22]
        self.viton_model_list = [None for i in range(len(self.garment_name_list))]
        self.lock = threading.Lock()

        self.load_all = Thread(target=self.load_all_models, args=())
        self.load_all.daemon = True
        self.load_all.start()

    def load_all_models(self):
        for i, garment_name in enumerate(self.garment_name_list):
            new_model = make_pix2pix_model(garment_name, 6, output_nc=4,ckpt_dir=self.ckpt_dir)
            if self.viton_model_list[i] is None:
                self.viton_model_list[i]= new_model

    def load_one_models(self, garment_name):
        new_model = make_pix2pix_model(garment_name, 6, output_nc=4)
        id = self.garment_name_list.index(garment_name)
        if self.viton_model_list[id] is None:
            self.viton_model_list[id]= new_model

    def switch_to_target_garment(self,garment_id):
        self.lock.acquire()
        print("Loading from CPU target garment id: ", garment_id)
        id = garment_id
        if self.viton_model_list[id] is None and id >= 0:
            print("Loading from disk target garment id: ", garment_id)
            self.load_one_models(self.garment_name_list[id])
        old_model = self.viton_model
        new_model=self.viton_model_list[id].to('cuda:0') if garment_id>=0 else None
        print("Finished")
        if self.viton_model is not None:
            del self.viton_model
        self.viton_model = new_model
        if old_model is not None:
            old_model = old_model.to('cpu')
            del old_model
            torch.cuda.empty_cache()
        self.lock.release()


    def set_target_garment(self, target_id):
        #new_model = make_pix2pix_model(ckpt_dict[target_id], 6, output_nc=4)
        #self.viton_model = self.viton_model_list[target_id]
        t = Thread(target=self.switch_to_target_garment, args=(target_id,))
        t.daemon = True
        t.start()

    def __call__(self, input_frame):
        if self.viton_model is None:
            return input_frame

        resolution = 512
        raw_image = input_frame

        smpl_data = self.smpl_regressor.forward(raw_image, True, size=1.45,
                                                roi_img_size=resolution)
        if len(smpl_data) < 3:
            return input_frame
        smpl_param, trans2roi, inv_trans2roi = smpl_data

        if smpl_param is None:
            return input_frame
        vertices = SMPL_Regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        height = raw_image.shape[0]
        width = raw_image.shape[1]

        v = vertices

        raw_vm = self.upper_body.render(v[0], height=height, width=width)

        raw_IUV = self.densepose_extractor.get_IUV(raw_image, isRGB=False)
        if raw_IUV is None:
            return input_frame
        dpi_img = IUV2SDP(raw_IUV)
        roi_dpi_img = cv2.warpAffine(dpi_img, trans2roi, (resolution, resolution), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

        roi_vm = cv2.warpAffine(raw_vm, trans2roi, (resolution, resolution), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0))

        vm_tensor = util.im2tensor(roi_vm) * 2.0 - 1.0
        vm_tensor = vm_tensor[:, [2, 1, 0], :, :]
        dp_tensor = util.im2tensor(roi_dpi_img) * 2.0 - 1.0
        self.lock.acquire()
        with torch.no_grad():
            if self.viton_model is not None:
                target_tensor = self.viton_model.forward(torch.cat([vm_tensor, dp_tensor], 1).cuda())
                self.lock.release()
            else:
                self.lock.release()
                return input_frame
        roi_target = util.tensor2im(target_tensor[0, [0, 1, 2], :, :], normalize=True, rgb=False)
        roi_alpha = (target_tensor[0, 3, :, :].clamp(min=0.0, max=1.0).cpu().numpy() * 255).astype(np.uint8)

        raw_target_img = cv2.warpAffine(roi_target, inv_trans2roi, (raw_image.shape[1], raw_image.shape[0]),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0))
        raw_alpha = cv2.warpAffine(roi_alpha, inv_trans2roi, (raw_image.shape[1], raw_image.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,))
        composed_img = naive_overlay_alpha(raw_image, raw_target_img, raw_alpha)
        return composed_img
