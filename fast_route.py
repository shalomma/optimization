import numpy as np
from newton import NewtonOptimizer


class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        self.start_x, self.start_y = start_x, start_y
        self.finish_x, self.finish_y = finish_x, finish_y
        self.velocities = velocities
        self.d = (self.finish_y - self.start_y) / len(velocities)
        self.dim = len(velocities) - 1

    def __call__(self, x):
        x = np.concatenate((x, [self.finish_x]))
        time = 0
        x_prev = self.start_x
        for i, x_i in enumerate(x):
            x_delta = x_i - x_prev
            hypotenuse = np.sqrt(x_delta ** 2 + self.d ** 2)
            time += hypotenuse / self.velocities[i]
            x_prev = x_i
        return time

    def grad(self, x):
        x = np.concatenate((x, [self.finish_x]))
        x_prev = self.start_x
        r = np.zeros(self.dim + 1)
        for i, x_i in enumerate(x):
            x_delta = x_i - x_prev
            hypotenuse = np.sqrt(x_delta ** 2 + self.d ** 2)
            r[i] = x_delta / (self.velocities[i] * hypotenuse)
            x_prev = x_i
        return r[:self.dim] - r[1:self.dim + 1]

    def hessian(self, x):
        x = np.concatenate((x, [self.finish_x]))
        x_prev = self.start_x
        p = np.zeros(self.dim + 1)
        for i, x_i in enumerate(x):
            x_delta = x_i - x_prev
            hypotenuse = np.sqrt(x_delta ** 2 + self.d ** 2)
            p[i] = self.d ** 2 / (self.velocities[i] * (hypotenuse ** 3))
        h = np.zeros((self.dim, self.dim))
        h[range(1, self.dim), range(0, self.dim - 1)] = - p[1:self.dim]
        h = h.T + h
        h[range(self.dim), range(self.dim)] = p[:self.dim] + p[1:self.dim + 1]
        return h


def find_fast_route(objective, init, alpha=1, threshold=1e-3, max_iters=1e3):
    opt = NewtonOptimizer(objective, alpha=alpha, init=init)
    opt.optimize(threshold, max_iters)
    return func(opt.x), opt.x, opt.num_iters


if __name__ == '__main__':
    velocities_ = np.array([2, 4, 5, 1, 1])
    func = FastRoute(start_x=0, start_y=0,
                     finish_x=100, finish_y=100,
                     velocities=velocities_)

    x0 = np.array([1, 2, 3, 40])

    fx, x, i = find_fast_route(func, x0, alpha=0.01, threshold=1e-14)

    dfx = func.grad(x)
    hx = func.hessian(x)
