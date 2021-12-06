from mapleDNNsat.dnn.dnnv_nn.visitors import OperationVisitor
import pdb
import onnx
from onnx import numpy_helper
from ..config import config
import numpy as np
from ..util import warning
from .milp_solver import Gurobi, SCIP
from ..types import Interval


solvers = {
    'scip': SCIP,
    'gurobi': Gurobi,
}


class ONNXBlaster:
    def __init__(self, dnn, property, solver=None):
        self.dnn = dnn
        self.property = property
        self.bounds = {}
        if solver is None:
            self.solver = SCIP() if config.scip else Gurobi()
        else:
            assert solver in solvers
            self.solver = solvers[solver]()
        self.interval_solution = None
        self.input_vars = []
        self.output_vars = []

    def build(self):
        layers = self.dnn.as_layers()
        assert len(layers) > 0
        vars, intervals = self._mk_input()
        for it, layer in enumerate(layers):
            is_output_layer = it + 1 == len(layers)
            vars, intervals = self._mk_fc(
                vars, intervals, layer, is_output=is_output_layer)
            if not is_output_layer:
                vars, intervals, the_pruned = self._mk_relu(vars, intervals, layer)
        self._mk_output(vars, intervals)
        self._mk_objective()

    ##Tony
    def get_pruned_num(self):
        layers = self.dnn.as_layers()
        assert len(layers) > 0
        vars, intervals = self._mk_input()
        tot_pruned = 0
        for it, layer in enumerate(layers):
            is_output_layer = it + 1 == len(layers)
            vars, intervals = self._mk_fc(
                vars, intervals, layer, is_output=is_output_layer)
            if not is_output_layer:
                vars, intervals, the_pruned = self._mk_relu(vars, intervals, layer)
                tot_pruned += the_pruned
        return tot_pruned
    ###        

    def _mk_objective(self):
        expr = self.solver.mk_expr()
        for var in self.input_vars:
            expr += var
        self.solver.set_objectve(expr, mode='max')

    def _mk_fc(self, input_vars, input_intervals, layer, out_var_prefix='x', is_output=False):
        out_vars = []
        out_intervals = [Interval() for _ in range(layer.out_shape)]
        for it in range(layer.out_shape):
            expr = self.solver.mk_expr()
            for jt in range(layer.in_shape):
                expr += layer.weights[jt][it] * input_vars[jt]
                out_intervals[it] += layer.weights[jt][it] * \
                    input_intervals[jt]
            expr += layer.biases[it]
            out_intervals[it] += layer.biases[it]
            var_name = f"{out_var_prefix}_{layer.depth}_{it}" if not is_output else f"out_{it}"
            lb, ub = out_intervals[it].low, out_intervals[it].upper
            out_vars.append(
                self.solver.mk_var(
                    name=var_name,
                    dtype='real',
                    lb=lb,
                    ub=ub,
                )
            )
            cons = self.solver.mk_cons(out_vars[-1], '==', expr)
        return out_vars, out_intervals

    def _mk_relu(self, input_vars, input_intervals, layer, out_var_prefix='y', relu_var_prefix='a'):
        out_vars = []
        out_intervals = [val.relu() for val in input_intervals]
        num_pruned = 0
        #y = relu(x)
        # Compute Y variables
        for it in range(layer.out_shape):
            x = input_vars[it]
            lb, ub = input_intervals[it].low, input_intervals[it].upper
            if lb >= 0:
                out_vars.append(x)
                num_pruned += 1
            elif ub <= 0:
                out_vars.append(0)
                num_pruned += 1
            else:
                a = self.solver.mk_var(
                    name=f'{relu_var_prefix}_{layer.depth}_{it}', dtype='bool')
                y = self.solver.mk_var(
                    name=f'{out_var_prefix}_{layer.depth}_{it}',
                    dtype='real',
                    lb=max(lb, 0.0),
                    ub=max(ub, 0.0),
                )
                out_vars.append(y)
                # TODO -- convert to indicator constraints
                cons = self.solver.mk_cons(y, '<=', x - lb * (1 - a))
                cons = self.solver.mk_cons(y, '>=', x)
                cons = self.solver.mk_cons(y, '<=', ub * a)
                cons = self.solver.mk_cons(y, '>=', 0.0)
        return out_vars, out_intervals, num_pruned

    def _mk_input(self):
        vars = []
        intervals = []
        lb, ub = self.property.input_bounds()
        for it, (lb, ub) in enumerate(zip(lb, ub)):
            vars.append(
                self.solver.mk_var(
                    name=f'in_{it}',
                    dtype='real',
                    lb=lb,
                    ub=ub)
            )
            intervals.append(
                Interval(lb, ub)
            )
        self.input_vars = vars
        A, b = self.property.input_mat_constraints()
        for it, row in enumerate(A):
            expr = self.solver.mk_expr()
            assert len(row) == len(self.input_vars)
            for a, var in zip(row, self.input_vars):
                expr += a * var
            self.solver.mk_cons(expr, '<=', b[it])
        return vars, intervals

    def _mk_output(self, input_vars, input_intervals):
        self.output_vars = input_vars
        lb, ub = self.property.output_bounds()
        for it, (lb, ub) in enumerate(zip(lb, ub)):
            self.solver.mk_cons(input_vars[it], '>=', lb)
            self.solver.mk_cons(input_vars[it], '<=', ub)
        A, b = self.property.output_mat_constraints()
        for it, row in enumerate(A):
            expr = self.solver.mk_expr()
            assert len(row) == len(self.output_vars)
            for a, var in zip(row, self.output_vars):
                expr += a * var
            self.solver.mk_cons(expr, '<=', b[it])

    def solve(self):
        ret = self.solver.solve()
        assert 'result' in ret
        assert ret['result'].lower() in ['sat', 'unsat', 'unknown']
        if ret['result'].lower() == 'sat':
            assert 'model' in ret
        return ret
