from .base import Solver, Var, Constraint, Expr
from ...config import config
import pdb


class CVC5Var(Var):
    def __init__(self, name, type, var) -> None:
        super().__init__(name, type)
        self.var = var
        self.expr_class = CVC5Expr


class CVC5Expr(Expr):
    def __init__(self, init_expr, init_expr_str) -> None:
        super().__init__(init_expr=init_expr, init_expr_str=init_expr_str)
        # self.expr = init_expr if init_expr is not None else pyCVC5opt.Expr()


class CVC5Constraint(Constraint):
    def __init__(self, lhs, op, rhs) -> None:
        super().__init__(lhs, op, rhs)


class CVC5(Solver):
    def __init__(self):
        super().__init__()
        # self.model = pyCVC5opt.Model("milp")

    def mk_var(self, dtype, name, ub=10**6, lb=-10**6):
        super().mk_var(dtype, name)
        if dtype.lower() == 'real':
            vtype = 'C'
            core_var = self.model.addVar(vtype=vtype, lb=lb, ub=ub, name=name)
        elif dtype.lower() == 'bool':
            vtype = 'B'
            core_var = self.model.addVar(vtype=vtype, name=name)
        else:
            raise ValueError(f"Unsupported variable type: {dtype}")
        self.vars.append(CVC5Var(
            name=name, type=dtype, var=core_var
        ))
        return self.vars[-1]

    def mk_expr(self):
        super().mk_expr()
        return CVC5Expr(init_expr=pyCVC5opt.Expr(), init_expr_str="")

    def write(self, file="model.lp"):
        self.model.write(file)

    def mk_cons(self, lhs, op, rhs):
        super().mk_cons(lhs, op, rhs)
        self.cons.append(CVC5Constraint(lhs, op, rhs))

        if isinstance(lhs, CVC5Expr):
            lhs = lhs.expr
        if isinstance(rhs, CVC5Expr):
            rhs = rhs.expr
        if isinstance(lhs, CVC5Var):
            lhs = lhs.var
        if isinstance(rhs, CVC5Var):
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
        # self.model.setPresolve(pyCVC5opt.CVC5_PARAMSETTING.OFF)
        # self.model.setParam('numerics/feastol', config.epsilon_CVC5)
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

    def write(self, file='model_CVC5.lp'):
        self.model.writeProblem(file)
