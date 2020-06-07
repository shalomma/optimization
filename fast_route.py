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
            x_prev = x_i
        h = np.zeros((self.dim, self.dim))
        h[range(1, self.dim), range(0, self.dim - 1)] = - p[1:self.dim]
        h = h.T + h
        h[range(self.dim), range(self.dim)] = p[:self.dim] + p[1:self.dim + 1]
        return h


def find_fast_route(objective, init, alpha=1, threshold=1e-3, max_iters=1e3):
    opt = NewtonOptimizer(objective, alpha=alpha, init=init)
    return opt.optimize(threshold, max_iters)


def find_alpha(start_x, start_y, finish_x, finish_y, num_layers):
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
