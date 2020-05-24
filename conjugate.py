class ConjGradOptimizer:
    def __init__(self, func, init):
        self.func = func
        self.x = init
        self.d = - self.func.grad(init)
        self.alpha = None
        self.x_prev = None

    def update_grad(self):
        return self.func.grad(self.x), self.func.grad(self.x_prev)

    def update_dir(self):
        g, g_prev = self.update_grad()
        beta = g.dot(g) / g_prev.dot(g_prev)
        self.d = - self.func.grad(self.x) + beta * self.d

    def update_alpha(self):
        # TODO: exact line search in the d direction
        pass

    def update_x(self):
        self.x_prev = self.x
        self.x += self.alpha * self.d

    def step(self):
        self.update_alpha()
        self.update_x()
        self.update_dir()

    def optimize(self):
        i = 0
        for i in range(10):  # TODO: understand the stopping criteria
            self.step()
        return self.func(self.x), self.x, i
