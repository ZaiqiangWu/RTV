import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--input_video', type=str, help='the path of input video file')
        self.parser.add_argument('--garment_name', type=str, default='example_garment', help='id of the target garment')
        self.initialized=True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


