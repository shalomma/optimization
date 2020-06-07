import numpy as np
from newton import NewtonOptimizer


class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        self.start_x, self.finish_x = start_x, finish_x
        self.velocities = velocities
        self.d = (finish_y - start_y) / len(velocities)
        self.dim = len(velocities) - 1

    def get_lengths(self, x):
        x = np.concatenate(([self.start_x], x, [self.finish_x]))
        x_deltas = x[1:] - x[:-1]
        d = np.array([self.d] * (self.dim + 1))
        hypotenuses = np.sqrt(x_deltas ** 2 + d ** 2)
        return x_deltas, hypotenuses

    def __call__(self, x):
        _, hypotenuses = self.get_lengths(x)
        times = hypotenuses / self.velocities
        return times.sum()

    def grad(self, x):
        """
        using a temp array 'r'
        r[k] = (x[k] - x[k-1]) / (v[k] * ((x[k] - x[k-1])^2 + d^2)^0.5)
        to construct the gradient vector as follows
        grad[k] = r[k] - r[k+1]
        """
        x_deltas, hypotenuses = self.get_lengths(x)
        r = x_deltas / (self.velocities * hypotenuses)
        return r[:self.dim] - r[1:self.dim + 1]

    def hessian(self, x):
        """
        using a temp array 'p'
        p[k] = d^2 / (v[k] * ((x[k] - x[k-1])^2 + d^2)^1.5)
        to construct the hessian matrix as follows
        h[k,k] = p[k] + p[k+1]
        h[k, k-1] = -p[k]
        and the rest of the elements are zeros
        """
        x_deltas, hypotenuses = self.get_lengths(x)
        p = self.d ** 2 / (self.velocities * hypotenuses ** 3)
        h = np.zeros((self.dim, self.dim))
        h[range(1, self.dim), range(0, self.dim - 1)] = - p[1:self.dim]
        h = h.T + h
        h[range(self.dim), range(self.dim)] = p[:self.dim] + p[1:self.dim + 1]
        return h


def find_fast_route(objective, init, alpha=1, threshold=1e-3, max_iters=1e3):
    opt = NewtonOptimizer(objective, alpha=alpha, init=init)
    return opt.optimize(threshold, max_iters)


def find_alpha(start_x, start_y, finish_x, finish_y, num_layers):
    """
    alpha should be inversely proportional to the average diagonal.
    """
    delta_x = finish_x - start_x
    delta_y = finish_y - start_y
    hypotenuse = np.sqrt(delta_x ** 2 + delta_y ** 2)
    avg_hypotenuse = hypotenuse / num_layers
    return min(1, 1 / 10 * avg_hypotenuse)


if __name__ == '__main__':
    velocities_ = np.array([1.9, 0.6, 0.53, 2.3, 2.5, 1.2, 0.6])
    start_x_, start_y_ = 0.4, 0.3
    finish_x_, finish_y_ = 11.2, 12.3
    func = FastRoute(start_x_, start_y_, finish_x_, finish_y_, velocities_)
    x0 = np.array([1.2, 2.3, 3.4, 4.5, 4.3, 3.2])

    alpha_ = find_alpha(start_x_, start_y_, finish_x_, finish_y_, len(velocities_))

    fx_, x_, i_ = find_fast_route(func, x0, alpha=alpha_, threshold=1e-5, max_iters=1e4)

    dfx_ = func.grad(x_)
    hx_ = func.hessian(x_)
