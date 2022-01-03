#!/usr/bin/env python3
import sys
import traceback
import pdb
import cProfile
import os
import tempfile
import torch
from .config import config
from .mlp import mk_dnn
from .vnnlib import mk_find_needle
import numpy as np


class ONNX_Fuzz:
    def __init__(self) -> None:
        #Parameters
        self.input_size = None
        self.output_size = None
        self.depth = None
        self.width_of_layer = None
        self.input_range = None

        self.dnn = None
        self.filename = None
    def prepare_parameters(self,in_size,out_size,depth,width,in_range):
        self.input_size = in_size
        self.output_size = out_size
        self.depth = depth
        self.width_of_layer = width
        self.input_range = in_range

    def prepare_nn(self,in_size,out_size,depth,width,in_range):
        self.prepare_parameters(in_size,out_size,depth,width,in_range)
        layers = [self.width_of_layer for _ in range(self.depth)]
        dnn = mk_dnn(n_features=self.input_size, n_outs=self.output_size, layers=layers)
        self.dnn = dnn

    def write_onnx(self,filename):
        self.filename = filename
        torch.onnx.export( self.dnn, f=self.filename, args=torch.Tensor([0] * self.dnn.in_shape),
        export_params=True, do_constant_folding=True,)

def mk_onnx(in_size, out_size,depth,width_layer,range):
    the_onnx = ONNX_Fuzz()
    the_onnx.prepare_nn(in_size, out_size,depth,width_layer,range)
    return the_onnx


    
