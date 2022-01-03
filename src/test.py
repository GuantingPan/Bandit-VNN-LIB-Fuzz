import os,subprocess
import time
if __name__ == '__main__':
    strat_time = time.time()
    try:
        subprocess.call('dnnf --vnnlib updated.vnnlib --network N updated.onnx',shell=True,timeout=1)
        print((time.time() - strat_time))
    except:
        print((time.time() - strat_time))
        print('hei')