class ConjGradOptimizer:
    def __init__(self, func, init):
        self.func = func
        self.x = init
        self.g = self.func.grad(init)
        self.g_prev = None
        self.d = - self.g
        self.dim = len(init)
        self.alpha = None
        self.x_prev = init

    def update_grad(self):
        return self.func.grad(self.x), self.func.grad(self.x_prev)

    def update_dir(self):
        beta = self.g.dot(self.g) / self.g_prev.dot(self.g_prev)
        return - self.g + beta * self.d

    def update_alpha(self):
        return - self.g.T.dot(self.d) / self.d.T.dot(self.func.Q).dot(self.d)

    def update_x(self):
        self.x_prev = self.x
        return self.x + self.alpha * self.d

    def step(self):
        self.alpha = self.update_alpha()
        self.x = self.update_x()
        self.g, self.g_prev = self.update_grad()
        self.d = self.update_dir()

    def optimize(self):
        i = 0
        for i in range(self.dim):
            self.step()
        return self.func(self.x), self.x, i
