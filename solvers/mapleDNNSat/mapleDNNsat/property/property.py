import pdb
from .dnnv_properties import parse
from functools import partial
from .dnnv_reductions import IOPolytopeReduction, HalfspacePolytope
from ..config import config
import numpy as np
reduction = partial(IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)


class Property:
    def __init__(self, file_path=None, property=None) -> None:
        self.path = file_path
        self.property = property
        self.reduction = partial(
            IOPolytopeReduction, HalfspacePolytope, HalfspacePolytope)

    def parse(self):
        self.property = parse(self.path, format='vnnlib')

    def concretize(self, dnn):
        _dnn = {'N': dnn.dnn}
        self.property.concretize(**_dnn)

    def disjunction(self):
        for subproperty in self.reducer():
            yield Property(property=subproperty)

    def num_subproperties(self):
        ret = 0
        for subproperty in self.reducer():
            ret += 1
        return ret

    def scale_acas(self, vec):
        # x_min = np.array([[1500.0, -0.06, 3.10, 980.0, 960.0]])
        # x_max = np.array([[1800.0, 0.06, 3.141593, 1200.0, 1200.0]])
        x_mean = np.array([[1.9791091e04, 0.0, 0.0, 650.0, 600.0]])
        x_range = np.array(
            [[60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]])

        return np.array((vec - x_mean) / x_range).flatten()

    def input_bounds(self):
        lb, ub = self.property.input_constraint.as_bounds()
        return lb, ub

    def input_mat_constraints(self):
        A_in, b_in = self.property.input_constraint.as_matrix_inequality()
        return A_in, b_in

    def output_mat_constraints(self):
        A_out, b_out = self.property.output_constraint.as_matrix_inequality(
            include_bounds=True)
        return A_out, b_out

    def output_bounds(self):
        lb, ub = self.property.output_constraint.as_bounds()
        return lb, ub

    def input_matrix_constraints(self):
        pass

    def output_matrix_constraints(self):
        pass

    def reducer(self):
        for subproperty in self.reduction(
                reduction_error=Exception
        ).reduce_property(~self.property):
            yield subproperty
