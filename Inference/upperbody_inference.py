import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..")))

from Inference.base_options import BaseOptions
from util.multithread_video_loader import MultithreadVideoLoader
from util.multithread_video_writer import MultithreadVideoWriter
from tqdm import tqdm
from VITON.viton_upperbody import FrameProcessor

def process_video(video_path, garment_name):
    video_loader = MultithreadVideoLoader(video_path,max_height=1024)
    video_writer = MultithreadVideoWriter(outvid='./output.mp4',fps=video_loader.get_fps())
    frame_processor = FrameProcessor([garment_name,],ckpt_dir='./checkpoints/')
    frame_processor.set_target_garment(0)
    for i in tqdm(range(len(video_loader))):
        frame = video_loader.cap()
        result = frame_processor(frame)
        video_writer.append(result)
    video_writer.make_video()
    video_writer.close()

if __name__ == '__main__':
    opts = BaseOptions()
    opt = opts.parse()
    video_path = opt.input_video
    garment_name = opt.garment_name
    process_video(video_path, garment_name)


