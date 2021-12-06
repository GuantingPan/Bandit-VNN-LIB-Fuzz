import torch.nn as nn
import torch.nn.functional as F
import torch
import onnx
import tempfile
from onnxsim import simplify


def pretty_vector(vec, decimals=3):
    ret = '['
    vec = vec.flatten()
    for val in vec:
        val = val.flatten()
        val = val.detach().numpy()
        assert val.shape == (1,), f"{val.shape}"
        val = float(val)
        ret += "%.3f" % round(val, decimals)
        ret += ','
    ret = ret[:-1] + ']'
    return ret


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
            print(rf"\t\Pre-Relu  = {pretty_vector(X)}")

            for jt, v in enumerate(X.detach().numpy().flatten()):
                model[f'a_{it}_{jt}'] = 1.0 if v >= 0 else 0.0
            X = F.relu(X)
            print(rf"\t\Post-Relu = {pretty_vector(X)}")
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


def mk_dnn(n_features=1, n_outs=1, layers=(2, 2)):
    dnn = MLP(in_shape=n_features, out_shape=n_outs, layer_sizes=layers)
    zero_sample = [0] * n_features
    zero_sample = torch.Tensor(zero_sample)
    # tmp = tempfile.NamedTemporaryFile()
    # torch.onnx.export(
    #     dnn,
    #     f=tmp.name, args=zero_sample,
    #     export_params=True,
    #     do_constant_folding=True,
    # )
    # onnx_model = onnx.load(tmp.name)
    # model, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model, file)
    return dnn
