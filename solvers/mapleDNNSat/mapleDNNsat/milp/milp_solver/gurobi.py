from gurobipy import Model, GRB, LinExpr
from .base import Solver, Var, Constraint, Expr
import pdb


class GurobiVar(Var):
    def __init__(self, name, type, var) -> None:
        super().__init__(name, type)
        self.var = var
        self.expr_class = GurobiExpr


class GurobiExpr(Expr):
    def __init__(self, init_expr, init_expr_str) -> None:
        super().__init__(init_expr=init_expr, init_expr_str=init_expr_str)
        self.expr = init_expr if init_expr is not None else LinExpr()


class GurobiConstraint(Constraint):
    def __init__(self, lhs, op, rhs) -> None:
        super().__init__(lhs, op, rhs)


def milp_callback(model, where):
    if where == GRB.Callback.MIP:
        obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
        obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
        if obj_bound > 0.01:
            model.terminate()
        if obj_best < -0.01:
            model.terminate()


class Gurobi(Solver):
    def __init__(self):
        super().__init__()
        self.model = Model("milp")

    def mk_var(self, dtype, name, ub=10**6, lb=-10**6):
        super().mk_var(dtype, name)
        if dtype.lower() == 'real':
            vtype = GRB.CONTINUOUS
            core_var = self.model.addVar(vtype=vtype, lb=lb, ub=ub, name=name)
        elif dtype.lower() == 'bool':
            vtype = GRB.BINARY
            core_var = self.model.addVar(vtype=vtype, name=name)
        self.vars.append(GurobiVar(name=name, type=dtype, var=core_var))
        return self.vars[-1]

    def mk_expr(self):
        super().mk_expr()
        return GurobiExpr(init_expr=LinExpr(), init_expr_str="")

    def write(self, file="model_gurobi.lp"):
        self.model.write(file)

    def mk_cons(self, lhs, op, rhs):
        super().mk_cons(lhs, op, rhs)
        self.cons.append(GurobiConstraint(lhs, op, rhs))
        if op == '==':
            op = GRB.EQUAL
        elif op == '<=':
            op = GRB.LESS_EQUAL
        elif op == '>=':
            op = GRB.GREATER_EQUAL
        else:
            raise ValueError
        if isinstance(lhs, GurobiExpr):
            lhs = lhs.expr
        if isinstance(rhs, GurobiExpr):
            rhs = rhs.expr
        if isinstance(lhs, GurobiVar):
            lhs = lhs.var
        if isinstance(rhs, GurobiVar):
            rhs = rhs.var
        self.model.addConstr(lhs, op, rhs)
        return self.cons[-1]

    def set_objectve(self, expr, mode):
        super().set_objectve(expr, mode)
        if mode == 'min':
            sense = GRB.MINIMIZE
        elif mode == 'max':
            sense = GRB.MAXIMIZE
        else:
            raise NotImplementedError
        self.model.setObjective(expr.expr, sense=sense)

    def solve(self):
        super().solve()
        self.model.setParam("SolutionLimit", 1)
        self.model.optimize()
        self.write()
        status = self.model.Status
        ret = {}
        # Optimal, Infeasable, Solution Limit, Killed by User
        if status not in [2, 3, 10, 11]:
            ret['result'] = 'ERROR'
        elif status in [2, 10]:  # Optimal, Solution Limit
            ret['result'] = 'SAT'
            ret['model'] = {
                'x': [var.x for var in self.model.getVars() if var.VarName.find('in_') != -1],
                'y': [var.x for var in self.model.getVars() if var.VarName.find('out_') != -1],
                'all': dict((var.VarName, var.x) for var in self.model.getVars()),
            }
        else:
            ret['result'] = 'UNSAT'
            ret['model'] = None
        return ret
