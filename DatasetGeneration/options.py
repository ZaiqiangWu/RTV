import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--video_path', type=str, help='the path of video')
        self.parser.add_argument('--mask_dir', type=str, help='the path of mask images')
        self.parser.add_argument('--dataset_name', type=str, default='example_dataset', help='name of the dataset')
        self.initialized=True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt