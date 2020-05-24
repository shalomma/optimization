class QuadraticFunction:
    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def __call__(self, x):
        return 0.5 * x.T * self.Q * x + self.b.T * x

    def grad(self, x):
        return self.Q * x

    def hessian(self, x):
        return self.Q
