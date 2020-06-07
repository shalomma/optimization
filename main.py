import numpy as np


def student_id():
    return 300312378, r'maayanshalom@mail.tau.ac.il'


# Part I
class QuadraticFunction:
    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def __call__(self, x):
        return 0.5 * x.T.dot(self.Q).dot(x) + self.b.T.dot(x)

    def grad(self, x):
        return 0.5 * (self.Q.T + self.Q).dot(x) + self.b

    def hessian(self, x):
        return 0.5 * (self.Q.T + self.Q)


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
        i = 0
        for i in range(1, int(max_iters + 1)):
            next_x, gx, hx = self.step(self.x)
            eps = np.linalg.norm(next_x - self.x, ord=2)
            self.x = next_x
            if eps < threshold:
                break
        return self.func(self.x), self.x, i


class ConjGradOptimizer:
    def __init__(self, func, init):
        self.func = func
        self.x = init
        self.g = self.func.grad(init)
        self.d = - self.g
        self.dim = len(init)
        self.alpha = None
        self.x_prev = None

    def step(self):
        self.alpha = self.update_alpha()
        self.x = self.update_x()
        self.g, prev_grad = self.update_grad()
        self.d = self.update_dir(prev_grad)
        return self.x, self.g, self.d, self.alpha

    def optimize(self):
        i = 0
        for i in range(1, self.dim + 1):
            self.step()
        return self.func(self.x), self.x, i

    def update_grad(self):
        return self.func.grad(self.x), self.func.grad(self.x_prev)

    def update_dir(self, prev_grad):
        beta = self.g.dot(self.g) / prev_grad.dot(prev_grad)
        return - self.g + beta * self.d

    def update_alpha(self):
        return - self.g.T.dot(self.d) / self.d.T.dot(self.func.Q).dot(self.d)

    def update_x(self):
        self.x_prev = self.x
        return self.x + self.alpha * self.d


# Part II
class FastRoute:
    def __init__(self, start_x, start_y, finish_x, finish_y, velocities):
        self.start_x, self.finish_x = start_x, finish_x
        self.velocities = velocities
        self.d = (finish_y - start_y) / len(velocities)
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
