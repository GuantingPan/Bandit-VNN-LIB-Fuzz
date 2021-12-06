#!/usr/bin/env python3
import sys
import traceback
import pdb
import cProfile
import os
import tempfile
import torch
from .config import config
from .util import die, warning
from .mlp import mk_dnn
from .vnnlib import mk_find_needle
import numpy as np

class Vnnlib_Fuzz:
    def __init__(self) -> None:
        self.onnx = None #This must be equal to an object from class ONNX_Fuzz() in onnx-fuzz.py
        self.in_tens = None
        self.out_tens = None

        self.vnnlib_str = None
        self.filename = None

    def prepare_SATvnnlib(self,the_dnn):
        self.onnx = the_dnn
        self.in_tens = torch.Tensor([np.random.uniform(-self.onnx.input_range, self.onnx.input_range) for _ in range(self.onnx.input_size)])
        self.out_tens = self.onnx.dnn(self.in_tens)
        self.vnnlib_str = mk_find_needle(in_tens=self.in_tens, out_tens=self.out_tens)

    def prepare_UNSATvnnlib(self,the_dnn):
        self.onnx = the_dnn
        self.in_tens = torch.Tensor([np.random.uniform(-self.onnx.input_range, self.onnx.input_range) for _ in range(self.onnx.input_size)])
        out_curr = self.onnx.dnn(self.in_tens)
        noise = torch.Tensor([np.random.uniform(-self.onnx.input_range, self.onnx.input_range) for _ in range(self.onnx.output_size)])
        self.out_tens = out_curr + noise
        self.vnnlib_str = mk_find_needle(in_tens=self.in_tens, out_tens=self.out_tens)

    def write_vnnlib(self,filename):
        self.filename = filename
        with open(self.filename, 'w') as vnn_file:
            vnn_file.write(self.vnnlib_str)

def mk_vnnlib(satisfiable,the_dnn):
    the_vnnlib = Vnnlib_Fuzz()
    if satisfiable:
        the_vnnlib.prepare_SATvnnlib(the_dnn=the_dnn)
    else:
        the_vnnlib.prepare_UNSATvnnlib(the_dnn=the_dnn)
    return the_vnnlib
    