import numpy as np


class NewtonOptimizer:
    def __init__(self, func, alpha, init):
        self.func = func
        self.alpha = alpha
        self.x = init
        self.num_iters = 0

    def step(self, x):
        gx = self.func.grad(x)
        hx = self.func.hessian(x)
        next_x = x - self.alpha * np.linalg.inv(hx).dot(gx)
        return next_x, gx, hx

    def optimize(self, threshold, max_iters):
        for i in range(int(max_iters)):
            next_x, gx, hx = self.step(self.x)
            if np.linalg.norm(next_x - self.x, ord=1) < threshold:
                self.num_iters = i
                break
            self.x = next_x
