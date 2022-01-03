import numpy as np
import torch
import pdb
import sys
import onnx
from onnxsim import simplify
import torch.nn as nn
import torch.nn.functional as F
import warnings
import traceback
import tempfile
from mapleDNNsat import Solver

warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=TracerWarning)


class MLP(nn.Module):
    def __init__(self, in_shape, out_shape, layer_sizes):
        super(MLP, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.layer_sizes = layer_sizes

        _in, _out = self.in_shape, None
        for it_layer, layer_size in enumerate(self.layer_sizes):
            _out = layer_size
            self.__setattr__(
                f'fc_{it_layer}',
                nn.Linear(_in, _out, bias=True)
            )
            _in = _out
        self.__setattr__('out', nn.Linear(_in, self.out_shape, bias=True))

    def forward(self, X):
        for it, _ in enumerate(self.layer_sizes):
            layer = self.__getattr__(f'fc_{it}')
            X = layer(X)
            X = F.relu(X)

        layer = self.__getattr__(f'out')
        X = self.out(X)
        return X

    def forward_with_debug_and_model(self, X):
        print(f"Feed Forward on X = {pretty_vector(X)}")
        model = {}
        for jt, v in enumerate(X.detach().numpy().flatten()):
            model[f'in_{jt}'] = v

        for it, size in enumerate(self.layer_sizes):
            print(f"\tLayer #{it}")

            layer = self.__getattr__(f'fc_{it}')
            print(f"\t\tWeights = {pretty_vector(layer.weight)}")
            print(f"\t\tBias    = {pretty_vector(layer.bias)}")

            X = layer(X)
            for jt, v in enumerate(X.detach().numpy().flatten()):
                model[f'x_{it}_{jt}'] = v
            print(rf"\t\tPre-Relu  = {pretty_vector(X)}")

            for jt, v in enumerate(X.detach().numpy().flatten()):
                model[f'a_{it}_{jt}'] = 1.0 if v >= 0 else 0.0
            X = F.relu(X)
            print(f"\t\tPost-Relu = {pretty_vector(X)}")
            for jt, v in enumerate(X.detach().numpy().flatten()):
                model[f'y_{it}_{jt}'] = v

        print(f"\tOut Layer")
        layer = self.__getattr__(f'out')
        print(f"\t\tWeights = {pretty_vector(layer.weight)}")
        print(f"\t\tBias    = {pretty_vector(layer.bias)}")
        X = self.out(X)
        print(f"Final Output = {pretty_vector(X)}")
        for jt, v in enumerate(X.detach().numpy().flatten()):
            model[f'out_{jt}'] = v

        return X, model


def mk_dnn(file='fuzz.onnx', n_features=1, n_outs=1, layers=(2, 2)):
    dnn = MLP(in_shape=n_features, out_shape=n_outs, layer_sizes=layers)
    zero_sample = [0] * n_features
    zero_sample = torch.Tensor(zero_sample)
    torch.onnx.export(
        dnn,
        f='fuzz.onnx', args=zero_sample,
        export_params=True,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(file)
    model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, file)
    return dnn


def mk_find_needle(in_tens, out_tens, in_eps=0.1, out_eps=10 ** -3):
    ret = ''
    for it in range(len(in_tens)):
        ret += f"(declare-const X_{it} Real)\n"
    ret += "\n"
    for it in range(len(out_tens)):
        ret += f"(declare-const Y_{it} Real)\n"
    ret += "\n"

    ret += '; Input Box\n'
    ret += '(assert (and \n'
    for it, val in enumerate(in_tens.flatten().detach().numpy()):
        ret += f'\t (>= X_{it} {val - in_eps}) (<= X_{it} {val + in_eps})\n'
    ret += '))\n\n'

    ret += "; Output Box\n"
    ret += "(assert (and \n"
    for it, val in enumerate(out_tens.flatten().detach().numpy()):
        ret += f'\t (>= Y_{it} {val - out_eps}) (<= Y_{it} {val + out_eps})\n'
    ret += "))\n"
    return ret


def run_maple(vnnlib_str, pytorch_dnn):
    vnn_file = tempfile.NamedTemporaryFile('w', suffix='.vnnlib')
    vnn_file.write(vnnlib_str)
    vnn_file.flush()
    onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx')
    torch.onnx.export(
        pytorch_dnn,
        f=onnx_file.name, args=torch.Tensor([0] * pytorch_dnn.in_shape),
        export_params=True,
        do_constant_folding=True,
    )
    ret = Solver().solve(property_file=vnn_file.name, dnn_file=onnx_file.name)

    pdb.set_trace()


def main():
    while True:
        def rng(n=1000): return np.random.randint(1, n)
        in_size = rng()
        out_size = rng()
        num_layers = rng()
        layers = [rng() for _ in range(num_layers)]
        dnn = mk_dnn(n_features=in_size, n_outs=out_size, layers=layers)
        in_tens = torch.Tensor([np.random.uniform(-5, 5)
                                for _ in range(in_size)])
        out_tens = dnn(torch.Tensor(
            [np.random.uniform(-5, 5) for _ in range(in_size)]))
        vnnlib = mk_find_needle(in_tens=in_tens, out_tens=out_tens)
        run_maple(vnnlib_str=vnnlib, pytorch_dnn=dnn)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
