class QuadraticFunction:
    def __init__(self, Q, b):
        self.Q = Q
        self.b = b

    def __call__(self, x):
        return 0.5 * x.T.dot(self.Q).dot(x) + self.b.T.dot(x)

    def grad(self, x):
        return (self.Q.T + self.Q).dot(x) + self.b

    def hessian(self, x):
        return self.Q.T + self.Q
