import argparse
import os
import pdb


class Config:
    def __init__(self, args) -> None:
        for arg in dir(args):
            if '__' != arg[:2]:
                self.__setattr__(arg, args.__getattribute__(arg))
        self.check_args()

    def check_args(self):
        pass


parser = argparse.ArgumentParser()

parser.add_argument("-n", "-net", "--net", "--network", "-dnn", "--dnn", "--deep-neural-network",
                    metavar="dnn_file",
                    action="store",
                    dest="dnn_file",
                    default='',
                    help="Output DNN File",
                    type=str,
                    )

parser.add_argument("-nrng",
                    metavar="n_rng",
                    action="store",
                    dest="n_rng",
                    default=5,
                    help="n_rng hyper parameter",
                    type=int,
                    )

parser.add_argument("-p", "-prop", "--prop", "--property",
                    metavar="vnnlib_file",
                    action="store",
                    dest="vnnlib_file",
                    default='',
                    help="Output Property File",
                    type=str,
                    )

parser.add_argument("-debug", "--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="debug tool"
                    )

parser.add_argument("-sat", "--sat",
                    action="store_true",
                    dest='sat',
                    default=False,
                    help="Generate SAT/UNSAT",
                    )

config_obj = Config(parser.parse_args())
