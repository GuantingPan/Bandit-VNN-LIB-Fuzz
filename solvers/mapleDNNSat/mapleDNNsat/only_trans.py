import os
import pdb
import cProfile
import pstats
from .config import config
from .property import Property
from .dnn import DNN
from .util import warning, die
from .milp import ONNXBlaster
from .milp.milp_solver import SCIP, Gurobi
from mapleDNNsat import dnn


class Trans:
    def __init__(self) -> None:
        self.dnn = None
        self.property = None
        self.model = None

    def prepare_benchmark(self, dnn_file, property_file):
        if not os.path.exists(property_file):
            raise FileNotFoundError("Could not find property file")
        if not os.path.exists(dnn_file):
            raise FileNotFoundError("Could not find DNN file")

        self.dnn = DNN(file_path=dnn_file)
        self.dnn.parse()

        self.property = Property(file_path=property_file)
        self.property.parse()
        self.property.concretize(dnn=self.dnn)

    def solve(self, dnn_file=None, property_file=None, milp_solver=None):
        #profiler = cProfile.Profile()
        #profiler.enable()
        self.prepare_benchmark(dnn_file=dnn_file, property_file=property_file)
        ret = self.solving_core(milp_solver)
        #profiler.disable()
        #stats = pstats.Stats(profiler).sort_stats('tottime')
        #stats.print_stats()
        return ret

    def solving_core(self, milp_solver):
        for subproperty in self.property.disjunction():
            self.model = ONNXBlaster(self.dnn, subproperty, solver=milp_solver)
            self.model.build()
            #ret = self.model.solve()
            #if ret['result'].lower() == 'sat':
            #    if config.validate_witness:
            #        x = ret['model']['x']
            #        y = ret['model']['y']
            #        if not self.check_witness(x, y, subproperty):
            #            ret['result'] = 'unsound'
            #    return ret
            #elif ret['result'].lower() == 'unsat':
            #    continue  # Solve the next independent problem
            #else:
            #    raise ValueError(f"Unexpected result: {ret['result']}")
        return 0

    def check_witness(self, x, y, subproperty):
        is_ok = True
        try:
            lb, ub = subproperty.input_bounds()
            for it, val in enumerate(x):
                if not (lb[it] <= val + config.epsilon and\
                        val - config.epsilon <= ub[it]):
                    warning(f"Input #{it} violates box: {lb[it]} <= {val} <= {ub[it]}")
                    is_ok = False
            lb, ub = subproperty.output_bounds()
            for it, val in enumerate(y):
                if not(lb[it] <= val + config.epsilon and\
                    val - config.epsilon <= ub[it]):
                    warning(f"Output #{it} violates box: {lb[it]} <= {val} <= {ub[it]}")
                    is_ok = False
            true_y = list(self.dnn(x).flatten())
            for it, (y, y_) in enumerate(zip(y, true_y)):
                if not (abs(y - y_) <= config.epsilon):
                    warning(f"Output #{it} Inconsistent with DNN: |{y} - {y_}| = {abs(y-y_)} > {config.epsilon}")
                    is_ok = False
        except AssertionError:
            warning('Unsound Result')
            is_ok = False
        return is_ok
