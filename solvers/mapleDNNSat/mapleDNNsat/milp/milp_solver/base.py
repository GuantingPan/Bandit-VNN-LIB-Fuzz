import pdb
import numpy as np
from ...util import is_numeric_literal


class Var:
    def __init__(self, name, type) -> None:
        self.name = name
        self.type = type
        self.var = None
        self.expr_class = None

    def as_expr(self):
        pass

    def __str__(self):
        return self.name

    def __mul__(self, other):
        if isinstance(other, Var):
            return self.expr_class(init_expr=self.var * other.var, init_expr_str=f'{self}*{other}')
        elif is_numeric_literal(other):
            return self.expr_class(init_expr=self.var * other, init_expr_str=f'{self}*{other}')
        elif isinstance(other, Expr):
            return self.expr_class(init_expr=self.var * other.expr, init_expr_str=f'{self}*{other}')
        else:
            raise ValueError(
                f"Unexpected sort: {type(other)} in variable multiplication")

    def __rmul__(self, other):
        if isinstance(other, Var):
            return self.expr_class(init_expr=other.var * self.var, init_expr_str=f'{other}*{self}')
        elif is_numeric_literal(other):
            return self.expr_class(init_expr=other * self.var, init_expr_str=f'{other}*{self}')
        elif isinstance(other, Expr):
            return self.expr_class(init_expr=other.expr * self.var, init_expr_str=f'{other}*{self}')
        else:
            raise ValueError(
                f"Unexpected sort: {type(other)} in variable multiplication")

    def __add__(self, other):
        if isinstance(other, Var):
            return self.expr_class(init_expr=self.var + other.var, init_expr_str=f'{self}+{other}')
        elif is_numeric_literal(other):
            return self.expr_class(init_expr=self.var + other, init_expr_str=f'{self}+{other}')
        elif isinstance(other, Expr):
            return self.expr_class(init_expr=self.var + other.expr, init_expr_str=f'{self}+{other}')
        else:
            raise ValueError(
                f"Unexpected sort: {type(other)} in variable multiplication")

    def __radd__(self, other):
        if isinstance(other, Var):
            return self.expr_class(init_expr=other.var - self.var, init_expr_str=f'{other}-{self}')
        elif is_numeric_literal(other):
            return self.expr_class(init_expr=other - self.var, init_expr_str=f'{other}-{self}')
        elif isinstance(other, Expr):
            return self.expr_class(init_expr=other.expr - self.var, init_expr_str=f'{other}-{self}')
        else:
            raise ValueError(
                f"Unexpected sort: {type(other)} in variable multiplication")

    def __sub__(self, other):
        if isinstance(other, Var):
            return self.expr_class(init_expr=self.var - other.var, init_expr_str=f'{self}-{other}')
        elif is_numeric_literal(other):
            return self.expr_class(init_expr=self.var - other, init_expr_str=f'{self}-{other}')
        elif isinstance(other, Expr):
            return self.expr_class(init_expr=self.var - other.expr, init_expr_str=f'{self}-{other}')
        else:
            raise ValueError(
                f"Unexpected sort: {type(other)} in variable multiplication")

    def __rsub__(self, other):
        if isinstance(other, Var):
            return self.expr_class(init_expr=other.var - self.var, init_expr_str=f'{other}-{self}')
        elif is_numeric_literal(other):
            return self.expr_class(init_expr=other - self.var, init_expr_str=f'{other}-{self}')
        elif isinstance(other, Expr):
            return self.expr_class(init_expr=other.expr - self.var, init_expr_str=f'{other}-{self}')
        else:
            raise ValueError(
                f"Unexpected sort: {type(other)} in variable multiplication")

    __repr__ = __str__


class Expr:
    def __init__(self, init_expr=None, init_expr_str='') -> None:
        self.expr = init_expr
        self.as_str = init_expr_str

    def __add__(self, other):
        self.as_str = str(self) + ' + ' + str(other)
        if isinstance(other, Expr):
            self.expr = self.expr + other.expr
        elif isinstance(other, Var):
            self.expr = self.expr + other.var
        elif is_numeric_literal(other):
            self.expr = self.expr + other
        else:
            raise ValueError(f"Unexpected arithmetic on type: {type(other)}")
        return self

    def __radd__(self, other):
        self.as_str = str(other) + ' + ' + str(self)
        if isinstance(other, Expr):
            self.expr = other.expr + self.expr
        elif isinstance(other, Var):
            self.expr = other.var + self.expr
        elif is_numeric_literal(other):
            self.expr = other + self.expr
        else:
            raise ValueError(f"Unexpected arithmetic on type: {type(other)}")
        return self

    def __mul__(self, other):
        self.as_str = str(self) + ' * ' + str(other)
        if isinstance(other, Expr):
            self.expr = self.expr * other.expr
        elif isinstance(other, Var):
            self.expr = self.expr * other.var
        elif is_numeric_literal(other):
            self.expr = self.expr * other
        else:
            raise ValueError(f"Unexpected arithmetic on type: {type(other)}")
        return self

    def __rmul__(self, other):
        self.as_str = str(other) + ' * ' + str(self)
        if isinstance(other, Expr):
            self.expr = other.expr * self.expr
        elif isinstance(other, Var):
            self.expr = other.var * self
        elif is_numeric_literal(other):
            self.expr = other * self.expr
        else:
            raise ValueError(f"Unexpected arithmetic on type: {type(other)}")
        return self

    def __str__(self) -> str:
        it = 0
        for it, c in enumerate(self.as_str):
            if c != '+' and c != ' ':
                return self.as_str[it:]
        return self.as_str
    __repr__ = __str__


class Constraint:
    def __init__(self, lhs, op, rhs) -> None:
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    def __str__(self) -> str:
        return f'{self.lhs} {self.op} {self.rhs}'
    __repr__ = __str__


class Solver:
    def __init__(self):
        self.vars = []
        self.cons = []

    def mk_var(self, dtype, name):
        assert dtype.lower() in ['real', 'bool']

    def mk_cons(self, lhs, op, rhs):
        assert op in ['<', '<=', '==', '>', '>=']

    def mk_expr(self):
        pass

    def solve(self):
        pass

    def set_objectve(self, expr, mode='max'):
        assert mode in ['min', 'max']
