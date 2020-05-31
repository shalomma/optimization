import numpy as np


class NewtonOptimizer:
    def __init__(self, func, alpha, init):
        self.func = func
        self.alpha = alpha
        self.x = init

    def step(self, x):
        gx = self.func.grad(x)
        hx = self.func.hessian(x)
        next_x = x - self.alpha * np.linalg.inv(hx).dot(gx)
        return next_x, gx, hx

    def optimize(self, threshold, max_iters):
        for i in range(1, int(max_iters + 1)):
            next_x, gx, hx = self.step(self.x)
            eps = np.linalg.norm(next_x - self.x, ord=2)
            self.x = next_x
            if eps < threshold:
                break
        return self.func(self.x), self.x, i
