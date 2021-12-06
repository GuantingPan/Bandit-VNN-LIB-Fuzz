from .base import Solver, Var, Constraint, Expr
from ...config import config
import pdb
from pysmt.shortcuts import Symbol, is_sat, get_model
from pysmt.typing import BOOL, INT, REAL


class SMTVar(Var):
    def __init__(self, name, type, var) -> None:
        super().__init__(name, type)
        self.var = var
        self.expr_class = SMTExpr


class SMTExpr(Expr):
    def __init__(self, init_expr, init_expr_str) -> None:
        super().__init__(init_expr=init_expr, init_expr_str=init_expr_str)
        # self.expr = init_expr if init_expr is not None else pySMTopt.Expr()


class SMTConstraint(Constraint):
    def __init__(self, lhs, op, rhs) -> None:
        super().__init__(lhs, op, rhs)


class SMT(Solver):
    def __init__(self):
        super().__init__()
        # self.model = pySMTopt.Model("milp")

    def mk_var(self, dtype, name, ub=10**6, lb=-10**6):
        super().mk_var(dtype, name)
        if dtype.lower() == 'real':
            vtype = REAL
            core_var = Symbol(name=name, typename=vtype)
        elif dtype.lower() == 'bool':
            vtype = BOOL
            core_var = Symbol(name=name, typename=vtype)
        else:
            raise ValueError(f"Unsupported variable type: {dtype}")

        self.vars.append(SMTVar(
            name=name, type=dtype, var=core_var
        ))
        return self.vars[-1]

    def mk_expr(self):
        super().mk_expr()
        return SMTExpr(init_expr=pySMTopt.Expr(), init_expr_str="")

    def write(self, file="model.lp"):
        self.model.write(file)

    def mk_cons(self, lhs, op, rhs):
        super().mk_cons(lhs, op, rhs)
        self.cons.append(SMTConstraint(lhs, op, rhs))

        if isinstance(lhs, SMTExpr):
            lhs = lhs.expr
        if isinstance(rhs, SMTExpr):
            rhs = rhs.expr
        if isinstance(lhs, SMTVar):
            lhs = lhs.var
        if isinstance(rhs, SMTVar):
            rhs = rhs.var

        if op == '==':
            self.model.addCons(lhs == rhs)
        elif op == '<=':
            self.model.addCons(lhs <= rhs)
        elif op == '>=':
            self.model.addCons(lhs >= rhs)
        else:
            raise ValueError
        return self.cons[-1]

    def solve(self):
        super().solve()
        # self.model.setPresolve(pySMTopt.SMT_PARAMSETTING.OFF)
        # self.model.setParam('numerics/feastol', config.epsilon_SMT)
        self.write()
        self.model.optimize()
        # self.model.setParam('limits/solutions', 1)
        # self.model.hideOutput(quiet=config.debug)
        ans = self.model.getStatus()
        ret = {}
        if ans not in ['optimal', 'infeasible']:
            ret['result'] = 'ERROR'
        elif ans == 'optimal':
            ret['result'] = 'SAT'
            in_vars = sorted([v for v in self.model.getVars() if v.name.find(
                'in_') != -1], key=lambda v: int(v.name.split('_')[-1]))
            out_vars = sorted([v for v in self.model.getVars() if v.name.find(
                'out_') != -1], key=lambda v: int(v.name.split('_')[-1]))
            ret['model'] = {}
            ret['model']['all'] = dict(
                (v.name, self.model.getVal(v)) for v in self.model.getVars())
            ret['model']['x'] = [self.model.getVal(v) for v in in_vars]
            ret['model']['y'] = [self.model.getVal(v) for v in out_vars]
        else:
            ret['result'] = 'UNSAT'
            ret['model'] = None
        return ret

    def set_objectve(self, expr, mode):
        super().set_objectve(expr, mode)
        if mode == 'min':
            sense = 'minimize'
        elif mode == 'max':
            sense = 'maximize'
        else:
            raise NotImplementedError
        self.model.setObjective(expr.expr, sense=sense)

    def write(self, file='model_SMT.lp'):
        self.model.writeProblem(file)
