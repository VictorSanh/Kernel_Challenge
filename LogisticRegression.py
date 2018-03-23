from scipy import optimize as op
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, lbda = 0.1):
        self.lbda = lbda

    def cost(self, theta, X, y):
        n_samples, d = X.shape
        Z = sigmoid(X.dot(theta))

        cost = -y.T.dot(np.log(Z)) - (1 - y).T.dot(np.log(1 - Z))
        cost = cost/n_samples
        cost += self.lbda/(2*n_samples)*np.dot(theta[1:], theta[1:])

        return cost


    def gradient(self, theta, X, y):
        n_samples, d = X.shape
        H = sigmoid(X.dot(theta))

        tmp = np.copy(theta)
        tmp[0] = 0
        regu = self.lbda*tmp/n_samples

        gradient = X.T.dot(H-y)
        gradient = gradient/n_samples
        gradient += regu

        return gradient


    def train(self, X, y):
        #Add intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta_init = np.zeros((X.shape[1], 1))

        result = op.minimize(fun = self.cost,
                             x0 = theta_init,
                             args = (X, y),
                             method = 'Newton-CG',
                             jac = self.gradient)
        self.theta = result.x


    def predict(self, X):
        #Add intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        Z = sigmoid(X.dot(self.theta))
        return np.round(Z).astype(int)


    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
