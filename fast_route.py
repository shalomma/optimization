import numpy as np


class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        self.start_x, self.start_y = start_x, start_y
        self.finish_x, self.finish_y = finish_x, finish_y
        self.velocities = velocities

    def __call__(self, x):
        x = np.concatenate((x, [self.finish_x]))
        d = float((self.finish_y - self.start_y) / len(x))
        time = 0
        x_prev = self.start_x
        for i, x_i in enumerate(x):
            x_delta = x - x_prev
            dist = np.sqrt(x_delta ** 2 + d ** 2)
            time += dist / self.velocities[i]
            x_prev = x_i
        return time

    def grad(self, x):
        pass

    def hessian(self, x):
        pass
