import argparse
import os

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--input', type=str, help='the path of input file')
        self.parser.add_argument('--output', type=str, help='the path of output file')
        self.parser.add_argument('--target_id', type=int, default=0, help='id of the target garment')
        self.parser.add_argument('--background_id', type=int, default=-1, help='id of the target garment, set it to -1 if you do not want to change the banckground')
        self.initialized=True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt


