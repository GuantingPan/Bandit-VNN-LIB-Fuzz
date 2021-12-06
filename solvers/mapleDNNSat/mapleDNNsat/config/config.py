import argparse
import os
import pdb


class Config:
    def __init__(self, args) -> None:
        for arg in dir(args):
            if '__' != arg[:2]:
                self.__setattr__(arg, args.__getattribute__(arg))
        self.check()

    def check(self):
        if not self.gurobi and not self.scip and not self.z3 and not self.cvc4 and not self.maplescip:
            self.maplescip = True


parser = argparse.ArgumentParser()

parser.add_argument("-n", "-net", "--net", "--network", "-dnn", "--dnn", "--deep-neural-network",
                    metavar="dnn",
                    action="store",
                    dest="dnn",
                    default='',
                    help="Input DNN",
                    type=str,
                    )

parser.add_argument("-p", "-prop", "--prop", "--property",
                    metavar="property",
                    action="store",
                    dest="property",
                    default='',
                    help="Input Property",
                    type=str,
                    )

parser.add_argument("-profile",
                    action="store_true",
                    dest="profile",
                    default=False,
                    help="Profile Tool"
                    )

parser.add_argument("-debug", "--debug",
                    action="store_true",
                    dest="debug",
                    default=False,
                    help="debug tool"
                    )


parser.add_argument("--epsilon",
                    metavar="epsilon",
                    action="store",
                    dest="epsilon",
                    default=10 ** -3,
                    help="Machine Percison For Soundness Checking",
                    type=float,
                    )


parser.add_argument("--epsilon-scip",
                    metavar="epsilon_scip",
                    action="store",
                    dest="epsilon_scip",
                    default=10 ** -3,
                    help="Machine Percison Epsilon for SCIP solver",
                    type=float,
                    )


parser.add_argument("--scip", '-scip',
                    action="store_true",
                    dest="scip",
                    default=False,
                    help="Use SCIP"
                    )

parser.add_argument("--gurobi", '-gurobi',
                    action="store_true",
                    dest="gurobi",
                    default=False,
                    help="Use Gurobi"
                    )


parser.add_argument("--cvc4", '-cvc4',
                    action="store_true",
                    dest="cvc4",
                    default=False,
                    help="Use cvc4"
                    )


parser.add_argument("--z3", '-z3',
                    action="store_true",
                    dest="z3",
                    default=False,
                    help="Use z3"
                    )


parser.add_argument("--maple-scip", '-mscip',
                    action="store_true",
                    dest="maplescip",
                    default=False,
                    help="Use maple-scip"
                    )


parser.add_argument("--validate-witness",
                    action="store_true",
                    dest="validate_witness",
                    default=True,
                    help="Validate SAT witnesses"
                    )

config_obj = Config(parser.parse_args())
