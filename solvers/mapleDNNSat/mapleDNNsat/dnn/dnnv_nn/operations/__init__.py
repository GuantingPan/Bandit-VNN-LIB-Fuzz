from .base import *
from .generator import *
from .math import *
from .nn import *
from .patterns import *
from .tensor import *

# TODO: enable isinstance checks
Activation: OperationPattern = Relu | Sigmoid | Tanh
