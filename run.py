#!/usr/bin/env python

import torch
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from model import HED

import argparse


torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='./network-bsds500.pytorch', help='model path')
parser.add_argument('--input', default='./images/fish.jpg', help='input image')
parser.add_argument('--output', default='./fish_edge_out.jpg', help='output path')


##########################################################


def test(model, opt):
    print("--------------Start Detecting---------------")
    # input=c_h_w
    # default input:	w= 480,h == 320
    print("--------------load image from:",opt.input,'--------------')
    tensorInput = torch.FloatTensor(numpy.array(PIL.Image.open(opt.input))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
    intWidth = tensorInput.size(2)
    intHeight = tensorInput.size(1)
    tensorInput = tensorInput.cuda().view(1, 3, intHeight, intWidth)
    tensorOutput = (model(tensorInput)[0, :, :, :].cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)
    PIL.Image.fromarray(tensorOutput).save(opt.output)
    print("--------------save image to:", opt.output, '--------------')



if __name__ == '__main__':

    opt = parser.parse_args()
    model = HED.HED()
    model.load_state_dict(torch.load(opt.model))
    print('--------------Load model from', opt.model,'--------------')
    model.cuda().eval()
    test(model,opt)

