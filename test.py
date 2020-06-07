from main import *
import numpy as np


if __name__ == '__main__':
    # Part I
    Q = np.array([[2, 1, 1], [4, 1, 3], [4, 1, 5]])
    b = np.array([4, 5, 4])
    func = QuadraticFunction(Q, b)
    x0 = np.array([2, 1, 1])

    opt = NewtonOptimizer(func, alpha=1, init=x0)
    fx_newton, x_newton, i_newton = opt.optimize(1e-10, 1000)
    dfx_newton = func.grad(x_newton)

    opt = ConjGradOptimizer(func, x0)
    fx_cg, x_cg, i_cg = opt.optimize()
    dfx_cg = func.grad(x_cg)

    # Part II
    velocities_ = np.array([1.9, 0.6, 0.53, 2.3, 2.5, 1.2, 0.6])
    start_x_, start_y_ = 0.4, 0.3
    finish_x_, finish_y_ = 11.2, 12.3
    func = FastRoute(start_x_, start_y_, finish_x_, finish_y_, velocities_)
    x0 = np.array([1.2, 2.3, 3.4, 4.5, 4.3, 3.2])

    alpha_ = find_alpha(start_x_, start_y_, finish_x_, finish_y_, len(velocities_))

    fx_, x_, i_ = find_fast_route(func, x0, alpha=alpha_, threshold=1e-5, max_iters=1e4)

    dfx_ = func.grad(x_)
    hx_ = func.hessian(x_)