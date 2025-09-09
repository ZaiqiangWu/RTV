import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))

import util.util
import util.util as util
import torch
#from util.video_loader import VideoLoader
from util.multithread_video_loader import MultithreadVideoLoader
from util.image2video import Image2VideoWriter
from util.image_caption import ImageCaption
import time
from SMPL.smpl_np import SMPLModel
from SMPL.smpl_regressor import SMPL_Regressor
from SMPL.pose_filter import PoseFilter, OfflineFilter
import glm
from OffscreenRenderer.flat_renderer import FlatRenderer
from SMPL.amputated_smpl import AmputatedSMPL
from SMPL.upperbody_smpl.UpperBody import UpperBodySMPL
from util.image_warp import zoom_in, zoom_out,shift_image_right, shift_image_down,rotate_image
#from Graphonomy.human_parser import HumanParser
#from Graphonomy.multiscale_cihp_human_parser import MultiscaleCihpHumanParser
#from Graphonomy.dataset_settings import dataset_settings
#cihp_label=dataset_settings['cihp']['label']
from tqdm import tqdm
from model.DensePose.densepose_extractor import DensePoseExtractor
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg
#from util.SlideWindowMaskRefine import slide_window_garment_mask_refine
import json
from DatasetGeneration.options import BaseOptions








def make_video_loader(source_path):
    source_dataset_new = MultithreadVideoLoader(source_path,max_height=3072)
    print("Length of source dataset: %d" % len(source_dataset_new))
    return source_dataset_new


def make_flat_renderer(height, width):
    texPath ='./assets/color_pattern/board_300x300.png'
    flat_render = FlatRenderer(height, width, texPath=texPath,back_black=True)
    return flat_render


def gen_dataset(source_path, dataset_name):
    video_loader = make_video_loader(source_path)
    smpl_regressor = SMPL_Regressor(use_bev=True,fix_body=True)
    densepose_extractor = DensePoseExtractor()



    flat_renderer = None
    pre_trans = None
    #human_parser = MultiscaleCihpHumanParser()
    upper_body=UpperBodySMPL()
    target_path = os.path.join('./PerGarmentDatasets', dataset_name)
    os.makedirs(target_path, exist_ok=True)
    resolution=1024

    height=None
    width=None

    for i in tqdm(range(len(video_loader))):
        raw_image = video_loader.cap()
        if raw_image is None:
            break
        height = raw_image.shape[0]
        width = raw_image.shape[1]
        new_height = 1024
        new_width = new_height * width // height
        resized_image = cv2.resize(raw_image, (new_width, new_height))

        smpl_param, trans2roi, inv_trans2roi = smpl_regressor.forward(raw_image, True, size=1.45,
                                                                      roi_img_size=resolution)  # 1.38

        # print(list(smpl_param.keys()))
        if smpl_param is None:
            continue

        triangles = smpl_param['smpl_face'].cpu().numpy().astype(np.int32)
        cam_trans = smpl_param['cam_trans']
        depth_order = torch.sort(cam_trans[:, 2].cpu(), descending=False).indices.numpy()
        vertices = smpl_param['verts_camed_org'][depth_order].cpu().numpy()
        if pre_trans is not None:
            cam_trans[0][0:2] = cam_trans[0][0:2] * 0.5 + pre_trans[0:2] * 0.5
            cam_trans[0][2] = cam_trans[0][2] * 0.01 + pre_trans[2] * 0.99
        pre_trans = cam_trans[0]
        vertices = smpl_regressor.get_raw_verts(smpl_param)
        vertices = torch.from_numpy(vertices).unsqueeze(0)

        v = vertices

        raw_vm = upper_body.render(v[0], height=new_height, width=new_width)
        # raw_vm = flat_renderer.render(v[0], uv/ratio, f.reshape(-1), np.array(model.to_list()), np.array(view.to_list()),
        #                              np.array(projection.to_list()))

        # rendered = ((rendered.astype(np.float32) + raw_image.astype(np.float32))/2).astype(np.uint8)


        #IUV = densepose_extractor.get_IUV(roi_img)
        #dp_img = IUV2UpperBodyImg(IUV)
        raw_IUV = densepose_extractor.get_IUV(resized_image,isRGB=False)
        if raw_IUV is None:
            continue
        #raw_dp_img = densepose_extractor.IUV2img(raw_IUV)
        #torso_leg_img = IUV2TorsoLeg(raw_IUV)






        #refined_raw_garment_mask_img = slide_window_garment_mask_refine(human_parser, num_window=3, raw_image=resized_image)
        refined_raw_garment_mask_img = human_parser.GetCihpGarmentMask(resized_image,isRGB=False)
        refined_raw_garment_mask_img = refined_raw_garment_mask_img.astype(np.uint8) * 255
        refined_raw_garment_mask_img_tmp = cv2.resize(refined_raw_garment_mask_img, (width, height))



        raw_garment = raw_image.copy()
        raw_garment[refined_raw_garment_mask_img_tmp<127] = 0
        roi_garment_img = cv2.warpAffine(raw_garment, trans2roi, (resolution, resolution),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))

        garment_path=os.path.join(target_path,str(i).zfill(5) + '_garment.jpg')
        vm_path=os.path.join(target_path,str(i).zfill(5) + '_vm.jpg')
        mask_path = os.path.join(target_path, str(i).zfill(5) + '_mask.png')
        iuv_path = os.path.join(target_path, str(i).zfill(5) + '_iuv.npy')
        trans2roi_path = os.path.join(target_path, str(i).zfill(5) + '_trans2roi.npy')
        cv2.imwrite(garment_path,roi_garment_img)
        cv2.imwrite(vm_path,raw_vm)
        cv2.imwrite(mask_path,refined_raw_garment_mask_img)
        np.save(iuv_path, raw_IUV)
        np.save(trans2roi_path, trans2roi)
    dataset_info = {
        "height": height,
        "width": width
    }
    with open(os.path.join(target_path,"dataset_info.json"), "w") as outfile:
        json.dump(dataset_info, outfile)
    video_loader.close()


def mask2img(mask):
    mask=mask.astype(np.uint8)
    mask=mask*255

    return mask





def process_video(v_path,dataset_name):
    gen_dataset(v_path, dataset_name)


if __name__ == '__main__':
    opts = BaseOptions()
    opt = opts.parse()
    video_path = opt.video_path
    mask_dir = opt.mask_dir
    dataset_name = opt.dataset_name
    process_video('./VideoDatasets/mgirl/01.mp4', dataset_name='mgirl_00')





