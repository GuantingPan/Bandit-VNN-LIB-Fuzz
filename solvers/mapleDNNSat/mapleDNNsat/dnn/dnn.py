from .dnnv_nn import parse
from pathlib import Path
import numpy as np
import pdb
import onnx
from onnx import numpy_helper


def onnxshape_to_intlist(onnxshape):
    """
    ONNX has its own wrapper for shapes. Our optimizer expects a list of ints.

    Arguments
    ---------
    onnxshape : TensorShapeProto

    Return
    ------
    output : list
        list of ints corresponding to onnxshape
    """
    result = list(
        map(lambda j: 1 if j.dim_value is None else int(j.dim_value), onnxshape.dim))

    # No shape means a single value
    if not result:
        return [1]

    # convert NCHW to NHWC
    if len(result) == 4:
        return [result[0], result[2], result[3], result[1]]

    return result


class Layer:
    def __init__(self, type, weights, biases, depth):
        assert len(weights.shape) == 2 and len(biases.shape) == 1
        assert weights.shape[1] == len(biases)
        self.type = type
        self.weights = weights
        self.biases = biases
        self.in_shape = weights.shape[0]
        self.out_shape = weights.shape[1]
        self.depth = depth

    # def __str__(self):
    #     return f"{"


class DNN:
    def __init__(self, file_path) -> None:
        self.path = file_path
        self.dnn = None

    def parse(self):
        #		self.dnn = parse(Path(self.path)).simplify()
        self.dnn = parse(Path(self.path))

    def onnx(self):
        return self.dnn.as_onnx()

    def tf(self):
        return self.dnn.as_tf()

    def as_layers(self):
        return self._mk_layers()

    def pytorch(self):
        return self.dnn.as_pytorch()

    def __call__(self, x):
        x = np.array(x).reshape(*self.dnn.input_shape)
        x = np.float32(x)
        return self.dnn(x)

    def _mk_layers(self):
        ret = []
        model = self.onnx()
        shape_map, constants_map, output_node_map, input_node_map, placeholdernames, activations = self.prepare_model(
            model)
        weights, bias = None, None
        depth = 0
        for val in model.graph.initializer:
            op = val.name.split('_')[0]
            constants = constants_map[val.name]
            if op == 'Reshape':
                continue
            assert op in ['Sub', 'MatMul', 'Gemm', 'Add']
            if op == 'Sub':
                # Not sure why this is always here
                assert np.sum(np.abs(constants)) == 0
            else:
                if op in ['MatMul', 'Gemm']:
                    assert weights is None
                    weights = constants
                if op == 'Add':
                    assert bias is None and weights is not None
                    bias = constants
                    ret.append(Layer('Relu', weights, bias, depth))
                    weights, bias = None, None
                    depth += 1
        return ret

    # From ERAN

    def prepare_model(self, model):
        """
        The constructor has produced a graph_def with the help of the functions graph_util.convert_variables_to_constants and graph_util.remove_training_nodes.
        translate() takes that graph_def, imports it, and translates it into two lists which then can be processed by an Optimzer object.

        Return
        ------
        (operation_types, operation_resources) : (list, list)
                A tuple with two lists, the first one has items of type str and the second one of type dict. In the first list the operation types are stored (like "Add", "MatMul", etc.).
                In the second list we store the resources (matrices, biases, etc.) for those operations. It is organised as follows: operation_resources[i][domain] has the resources related to
                operation_types[i] when analyzed with domain (domain is currently either 'deepzono' or 'deeppoly', as of 8/30/18)
        """
        shape_map = {}
        constants_map = {}
        output_node_map = {}
        input_node_map = {}
        activations = []
        for initial in model.graph.initializer:
            const = numpy_helper.to_array(initial).copy()
            constants_map[initial.name] = const
            shape_map[initial.name] = const.shape

        placeholdernames = []
        #print("graph ", model.graph.node)
        for node_input in model.graph.input:
            placeholdernames.append(node_input.name)
            if node_input.name not in shape_map:
                shape_map[node_input.name] = onnxshape_to_intlist(
                    node_input.type.tensor_type.shape)
                input_node_map[node_input.name] = node_input

        for node in model.graph.node:
            # print(node.op_type)
            output_node_map[node.output[0]] = node
            for node_input in node.input:
                input_node_map[node_input] = node
            if node.op_type == "Flatten":
                #shape_map[node.output[0]] = shape_map[node.input[0]]
                shape_map[node.output[0]] = [1, ] + \
                    [np.prod(shape_map[node.input[0]][1:]), ]
            elif node.op_type == "Constant":
                const = node.attribute
                const = numpy_helper.to_array(const[0].t).copy()
                constants_map[node.output[0]] = const
                shape_map[node.output[0]] = const.shape

            elif node.op_type in ["MatMul", "Gemm"]:
                transA = 0
                transB = 0
                for attribute in node.attribute:
                    if 'transA' == attribute.name:
                        transA = attribute.i
                    elif 'transB' == attribute.name:
                        transB = attribute.i
                input_shape_A = ([1] if len(shape_map[node.input[0]]) == 1 else [
                ]) + list(shape_map[node.input[0]])
                input_shape_B = list(
                    shape_map[node.input[1]]) + ([1] if len(shape_map[node.input[1]]) == 1 else [])
                M = input_shape_A[transA]
                N = input_shape_B[1 - transB]
                shape_map[node.output[0]] = [M, N]

            elif node.op_type in ["Add", "Sub", "Mul", "Div"]:
                shape_map[node.output[0]] = shape_map[node.input[0]]
                if node.input[0] in constants_map and node.input[1] in constants_map:
                    if node.op_type == "Add":
                        result = np.add(
                            constants_map[node.input[0]], constants_map[node.input[1]])
                    elif node.op_type == "Sub":
                        result = np.subtract(
                            constants_map[node.input[0]], constants_map[node.input[1]])
                    elif node.op_type == "Mul":
                        result = np.multiply(
                            constants_map[node.input[0]], constants_map[node.input[1]])
                    elif node.op_type == "Div":
                        result = np.divide(
                            constants_map[node.input[0]], constants_map[node.input[1]])
                    constants_map[node.output[0]] = result
            elif node.op_type in ["Conv", "MaxPool", "AveragePool"]:
                output_shape = []
                input_shape = shape_map[node.input[0]]

                require_kernel_shape = node.op_type in [
                    "MaxPool", "AveragePool"]
                if not require_kernel_shape:
                    filter_shape = shape_map[node.input[1]]
                    kernel_shape = filter_shape[1:-1]

                strides = [1, 1]
                padding = [0, 0, 0, 0]
                auto_pad = 'NOTSET'
                dilations = [1, 1]
                group = 1
                ceil_mode = 0
                for attribute in node.attribute:
                    if attribute.name == 'strides':
                        strides = attribute.ints
                    elif attribute.name == 'pads':
                        padding = attribute.ints
                    elif attribute.name == 'auto_pad':
                        auto_pad = attribute.s
                    elif attribute.name == 'kernel_shape':
                        kernel_shape = attribute.ints
                    elif attribute.name == 'dilations':
                        dilations = attribute.ints
                    elif attribute.name == 'group':
                        group = attribute.i
                    elif attribute.name == 'ceil_mode':
                        ceil_mode = attribute.i

                effective_kernel_shape = [
                    (kernel_shape[i] -
                     1) *
                    dilations[i] +
                    1 for i in range(
                        len(kernel_shape))]

                output_shape.append(input_shape[0])

                for i in range(len(kernel_shape)):
                    effective_input_size = input_shape[1 + i]
                    effective_input_size += padding[i]
                    effective_input_size += padding[i + len(kernel_shape)]
                    if ceil_mode == 1:
                        strided_kernel_positions = int(
                            np.ceil(
                                (effective_input_size -
                                 effective_kernel_shape[i]) /
                                float(
                                    strides[i])))
                    else:
                        strided_kernel_positions = int(
                            np.floor((effective_input_size - effective_kernel_shape[i]) / strides[i]))
                    output_shape.append(1 + strided_kernel_positions)

                if require_kernel_shape:
                    output_shape.append(input_shape[3])
                else:
                    output_shape.append(filter_shape[0])

                shape_map[node.output[0]] = output_shape
            elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "BatchNormalization", "LeakyRelu"]:
                activations.append(node.op_type)
                shape_map[node.output[0]] = shape_map[node.input[0]]

            # Gather is for the moment solely for shapes
            elif node.op_type == "Gather":
                axis = 0
                for attribute in node.attribute:
                    axis = attribute.i
                if node.input[0] in constants_map and node.input[1] in constants_map:
                    data = constants_map[node.input[0]]
                    indexes = constants_map[node.input[1]]
                    constants_map[node.output[0]] = np.take(
                        data, indexes, axis)

                if node.input[0] in shape_map and node.input[1] in shape_map:
                    r = len(shape_map[node.input[0]])
                    q = len(shape_map[node.input[1]])
                    out_rank = q + r - 1
                    if out_rank == 0:
                        shape_map[node.output[0]] = shape_map[node.input[1]]
                    else:
                        output_shape = []
                        for i in range(out_rank):
                            if i < axis:
                                output_shape.append(
                                    shape_map[node.input[0]][i])  # i < axis < r
                            elif i >= axis and i < axis + q:
                                output_shape.append(
                                    shape_map[node.input[0]][i - axis])  # i - axis < q
                            else:
                                # i < out_rank < q + r - 1
                                output_shape.append(
                                    shape_map[node.input[0]][i - q + 1])
                        shape_map[node.output[0]] = output_shape
            elif node.op_type == "Shape":
                if node.input[0] in shape_map:
                    constants_map[node.output[0]] = shape_map[node.input[0]]
                    shape_map[node.output[0]] = [len(shape_map[node.input[0]])]

            # elif node.op_type == "Cast":
                #shape_map[node.output[0]] = shape_map[node.input[0]]
                #print("CASTING ", node.input[0], shape_map[node.input[0]], shape_map[node.output[0]])

            elif node.op_type == "Reshape":
                #print("RESHAPE ", node.input, node.output)
                if node.input[1] in constants_map:
                    total = 1
                    replace_index = -1
                    for index in range(len(constants_map[node.input[1]])):
                        if constants_map[node.input[1]][index] == -1:
                            replace_index = index
                        else:
                            total *= constants_map[node.input[1]][index]

                    if replace_index != -1:
                        constants_map[node.input[1]][replace_index] = np.prod(
                            shape_map[node.input[0]]) / total

                    if len(constants_map[node.input[1]]) == 4:
                        shape_map[node.output[0]] = [constants_map[node.input[1]][0],
                                                     constants_map[node.input[1]][2],
                                                     constants_map[node.input[1]][3],
                                                     constants_map[node.input[1]][1]]
                    else:
                        shape_map[node.output[0]
                                  ] = constants_map[node.input[1]]

            elif node.op_type == "Unsqueeze":
                if node.input[0] in shape_map:
                    axis = node.attribute[0].ints
                    output_shape = list(shape_map[node.input[0]])
                    if node.input[0] in constants_map:
                        constants_map[node.output[0]
                                      ] = constants_map[node.input[0]]
                    for i in axis:
                        output_shape.insert(i, 1)
                        if node.input[0] in constants_map:
                            constants_map[node.output[0]] = np.expand_dims(
                                constants_map[node.output[0]], axis=i)
                    shape_map[node.output[0]] = output_shape

            elif node.op_type == "Concat":
                all_constant = True
                n_dim = len(shape_map[node.input[0]])
                if n_dim > 2:
                    axis = node.attribute[0].i
                else:
                    axis = node.attribute[0].i
                for node_input in node.input:
                    if node_input not in constants_map:
                        all_constant = False
                        break
                if all_constant:
                    constants_map[node.output[0]] = np.concatenate(
                        [constants_map[input] for input in node.input], axis=axis)
                all_shape_known = True
                for node_input in node.input:
                    if node_input not in shape_map:
                        all_shape_known = False
                        break
                assert all_shape_known, "Unknown shape for at least one node input!"
                new_axis_size = 0
                for node_input in node.input:
                    new_axis_size += shape_map[node_input][axis]
                shape_map[node.output[0]] = [shape_map[node.input[0]][i] if i !=
                                             axis else new_axis_size for i in range(len(shape_map[node.input[0]]))]
                if not all_constant:
                    assert axis == n_dim - \
                        1, "ELINA currently only supports concatenation on the channel dimension"

            elif node.op_type == "Tile":
                repeats = constants_map[node.input[1]]
                input_shape = list(shape_map[node.input[0]])
                assert len(repeats) == len(
                    input_shape), "Expecting one repeat factor per dimension"
                output_shape = [
                    factor * size for factor,
                    size in zip(
                        repeats,
                        input_shape)]
                shape_map[node.output[0]] = output_shape

                repeat_index = np.where(np.array(repeats) != 1)[0]
                assert len(
                    repeat_index) == 1, "ELINA backend currently only supports repeats for one dimension"
                repeat_index = repeat_index.item()
                assert repeat_index == 1, "ELINA backend currently only supports repeats for the first dimension"
                assert input_shape[0] == 1, "ELINA backend currently only supports repeats for dimensions of size 1"

            elif node.op_type == "Expand":
                if node.input[1] in constants_map:
                    if len(constants_map[node.input[1]]) == 4:
                        shape_map[node.output[0]] = [constants_map[node.input[1]][0],
                                                     constants_map[node.input[1]][2],
                                                     constants_map[node.input[1]][3],
                                                     constants_map[node.input[1]][1]]
                    else:
                        shape_map[node.output[0]
                                  ] = constants_map[node.input[1]]

                    result = np.zeros(
                        shape_map[node.output[0]]) + constants_map[node.input[0]]
                    constants_map[node.output[0]] = result
            elif node.op_type == "Pad":
                input_shape = np.array(shape_map[node.input[0]])
                for attribute in node.attribute:
                    if attribute.name == "pads":
                        padding = np.array(attribute.ints)
                    if attribute.name == "mode":
                        assert attribute.s == bytes(
                            b'constant'), "only zero padding supported"
                    if attribute.name == "value":
                        assert attribute.f == 0, "only zero padding supported"
                output_shape = np.copy(input_shape)
                input_dim = len(input_shape)
                assert len(padding) == 2 * input_dim
                for i in range(2, input_dim):  # only pad spatial dimensions
                    output_shape[i - 1] += padding[i] + padding[i + input_dim]
                shape_map[node.output[0]] = list(output_shape)
            else:
                assert 0, f"Operations of type {node.op_type} are not yet supported."
        return shape_map, constants_map, output_node_map, input_node_map, placeholdernames, activations
