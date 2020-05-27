import numpy as np


class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        self.start_x, self.start_y = start_x, start_y
        self.finish_x, self.finish_y = finish_x, finish_y
        self.velocities = velocities
        self.d = (self.finish_y - self.start_y) / len(velocities)

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
        r = []
        for i, x_i in enumerate(x):
            x_delta = x_i - x_prev
            hypotenuse = np.sqrt(x_delta ** 2 + self.d ** 2)
            r.append(x_delta / (self.velocities[i] * hypotenuse))
            x_prev = x_i
        g = []
        for i in range(len(x) - 1):
            g.append(r[i] + r[i + 1])
        return np.array(g)

    def hessian(self, x):
        pass


if __name__ == '__main__':
    velocities_ = np.array([1, 1, 1, 1, 1, 1, 1])
    func = FastRoute(0, 0, 7, 7, velocities_)
    x0 = np.array([1, 2, 3, 4, 5, 6])

    fx = func(x0)
    dfx = func.grad(x0)
