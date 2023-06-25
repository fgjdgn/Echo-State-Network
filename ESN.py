import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg

def train_and_test_ESN(data, train_len=2000, test_len=100, init_len=100, in_size=3, res_size=1000, a=0.7, reg=1e-8):
    # Generate ESN reservoir
    np.random.seed(42)
    Win = (np.random.rand(res_size, 1 + in_size) - 0.5) * 1
    W = np.random.rand(res_size, res_size) - 0.5
    rhoW = max(np.abs(linalg.eig(W)[0]))
    W *= 1.25 / rhoW

    # Allocate memory for the reservoir states matrix
    X = np.zeros((1 + in_size + res_size, train_len - init_len))
    Yt = data[None, init_len + 1:train_len + 1]

    x = np.zeros((res_size, 1))
    x = np.squeeze(x)
    for t in range(train_len):
        u = data[t]
        x = (1 - a) * x + a * np.tanh(np.dot(Win, np.hstack((1, u))) + np.dot(W, x))
        if t >= init_len:
            X[:, t - init_len] = np.hstack((1, u, x))

    # Use Ridge Regression to fit the target values
    X_T = X.T
    Wout = np.dot(np.dot(np.squeeze(data[None, init_len + 1:train_len + 1]).T, X_T),
                  linalg.inv(np.dot(X, X_T) + reg * np.eye(1 + in_size + res_size)))

    # Run the trained ESN in generative mode
    Y = np.zeros((out_size, test_len))
    u = data[train_len]
    for t in range(test_len):
        x = (1 - a) * x + a * np.tanh(np.dot(Win, np.hstack((1, u))) + np.dot(W, np.squeeze(x)))
        y = np.dot(Wout, np.hstack((1, u, x)))
        Y[:, t] = y
        u = y

    return Y

# Usage example:
data = # Your data here
output = train_and_test_ESN(data)
