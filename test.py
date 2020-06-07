import numpy as np
from quadratic import QuadraticFunction
from conjugate import ConjGradOptimizer
from newton import NewtonOptimizer


if __name__ == '__main__':
    Q = np.array([[2, 1, 1], [4, 1, 3], [4, 1, 5]])
    b = np.array([4, 5, 4])
    func = QuadraticFunction(Q, b)

    x0 = np.array([2, 1, 1])

    opt = NewtonOptimizer(func, alpha=1, init=x0)
    fx_newton, x_newton, i_newton = opt.optimize(1e-10, 1000)

    opt = ConjGradOptimizer(func, x0)
    fx_cg, x_cg, i_cg = opt.optimize()
