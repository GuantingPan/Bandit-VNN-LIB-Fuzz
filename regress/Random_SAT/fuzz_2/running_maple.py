from mapleDNNsat.dnn.dnnv_nn.converters import onnx
import unittest
import tempfile
import torch
import os
import pdb
from mapleDNNsat import Solver

loc = os.path.dirname(os.path.realpath(__file__))

class running2(unittest.TestCase):

    def run_maple(self, vnnlib_file, onnx_file, milp_solver='scip'):
        ret = Solver().solve(property_file=vnnlib_file,
                             dnn_file=onnx_file, milp_solver=milp_solver)
        self.assertEqual(ret['result'].lower(), 'sat')

    def test_fuzz_1(self): self.run_maple(
        vnnlib_file=f"{loc}/1.vnnlib", onnx_file=f"{loc}/1.onnx", milp_solver='scip')
    def test_fuzz_2(self): self.run_maple(
        vnnlib_file=f"{loc}/2.vnnlib", onnx_file=f"{loc}/2.onnx", milp_solver='scip')
    def test_fuzz_3(self): self.run_maple(
        vnnlib_file=f"{loc}/3.vnnlib", onnx_file=f"{loc}/3.onnx", milp_solver='scip')
    def test_fuzz_4(self): self.run_maple(
        vnnlib_file=f"{loc}/4.vnnlib", onnx_file=f"{loc}/4.onnx", milp_solver='scip')
    def test_fuzz_5(self): self.run_maple(
        vnnlib_file=f"{loc}/5.vnnlib", onnx_file=f"{loc}/5.onnx", milp_solver='scip')
    def test_fuzz_6(self): self.run_maple(
        vnnlib_file=f"{loc}/6.vnnlib", onnx_file=f"{loc}/6.onnx", milp_solver='scip')
    def test_fuzz_7(self): self.run_maple(
        vnnlib_file=f"{loc}/7.vnnlib", onnx_file=f"{loc}/7.onnx", milp_solver='scip')
    def test_fuzz_8(self): self.run_maple(
        vnnlib_file=f"{loc}/8.vnnlib", onnx_file=f"{loc}/8.onnx", milp_solver='scip')
    def test_fuzz_9(self): self.run_maple(
        vnnlib_file=f"{loc}/9.vnnlib", onnx_file=f"{loc}/9.onnx", milp_solver='scip')
    def test_fuzz_10(self): self.run_maple(
        vnnlib_file=f"{loc}/10.vnnlib", onnx_file=f"{loc}/10.onnx", milp_solver='scip')
    
    def test_fuzz_11(self): self.run_maple(
        vnnlib_file=f"{loc}/1.vnnlib", onnx_file=f"{loc}/1.onnx", milp_solver='gurobi')
    def test_fuzz_12(self): self.run_maple(
        vnnlib_file=f"{loc}/2.vnnlib", onnx_file=f"{loc}/2.onnx", milp_solver='gurobi')
    def test_fuzz_13(self): self.run_maple(
        vnnlib_file=f"{loc}/3.vnnlib", onnx_file=f"{loc}/3.onnx", milp_solver='gurobi')
    def test_fuzz_14(self): self.run_maple(
        vnnlib_file=f"{loc}/4.vnnlib", onnx_file=f"{loc}/4.onnx", milp_solver='gurobi')
    def test_fuzz_15(self): self.run_maple(
        vnnlib_file=f"{loc}/5.vnnlib", onnx_file=f"{loc}/5.onnx", milp_solver='gurobi')
    def test_fuzz_16(self): self.run_maple(
        vnnlib_file=f"{loc}/6.vnnlib", onnx_file=f"{loc}/6.onnx", milp_solver='gurobi')
    def test_fuzz_17(self): self.run_maple(
        vnnlib_file=f"{loc}/7.vnnlib", onnx_file=f"{loc}/7.onnx", milp_solver='gurobi')
    def test_fuzz_18(self): self.run_maple(
        vnnlib_file=f"{loc}/8.vnnlib", onnx_file=f"{loc}/8.onnx", milp_solver='gurobi')
    def test_fuzz_19(self): self.run_maple(
        vnnlib_file=f"{loc}/9.vnnlib", onnx_file=f"{loc}/9.onnx", milp_solver='gurobi')
    def test_fuzz_20(self): self.run_maple(
        vnnlib_file=f"{loc}/10.vnnlib", onnx_file=f"{loc}/10.onnx", milp_solver='gurobi')

if __name__ == '__main__':
    unittest.main()
