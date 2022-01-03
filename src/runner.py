import os

if __name__ == '__main__':
    os.system('dnnf --vnnlib ../regress/Random_SAT/fuzz_3/10.vnnlib'+' --network N ../regress/Random_SAT/fuzz_3/10.onnx')