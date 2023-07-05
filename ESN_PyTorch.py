import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class ESN(nn.Module):
    def __init__(self, reservoir_size=1000, learning_rate=0.5, spectral_radius=1.25, reg=1e-8, random_seed=42):
        super(ESN, self).__init__()
        random.seed(random_seed)
        self.reservoir_size = reservoir_size
        self.learning_rate = learning_rate
        self.spectral_radius = spectral_radius
        self.reg = reg
        self.Win = None
        self.W = None
        self.Wout = None
        self.x = None
        self.input_size = None
        self.output_size = None
        self.training_data = None

    def train(self, data, train_len, init_len):
        self.input_size = data.shape[1]
        self.output_size = data.shape[1]
        self.training_data = data

        # 初始化权重
        self.Win = torch.rand(self.reservoir_size, 1 + self.input_size, dtype=torch.double) - 0.5
        self.W = torch.rand(self.reservoir_size, self.reservoir_size, dtype=torch.double) - 0.5
        self.W *= self.spectral_radius / torch.max(torch.abs(torch.linalg.eig(self.W)[0]))

        X = torch.zeros((1 + self.input_size + self.reservoir_size, train_len - init_len), dtype=torch.double)
        Yt = torch.tensor(data[None, init_len + 1:train_len + 1])

        self.x = torch.zeros((self.reservoir_size, 1), dtype=torch.double)
        self.x = self.x.squeeze()

        for t in range(train_len):
            u = torch.tensor(data[t])
            # 更新隐藏状态
            self.x = (1 - self.learning_rate) * self.x + self.learning_rate * torch.tanh(
                torch.matmul(self.Win, torch.cat((torch.tensor([1.0]), u))) + torch.matmul(self.W, self.x)
            )
            if t >= init_len:
                X[:, t - init_len] = torch.cat((torch.tensor([1.0]), u, self.x))

        X_T = X.T
        # 计算输出权重
        self.Wout = torch.matmul(torch.matmul(torch.tensor(data[None, init_len + 1:train_len + 1].squeeze().T), X_T),
                                 torch.inverse(torch.matmul(X, X_T) + self.reg * torch.eye(1 + self.input_size + self.reservoir_size)))

    def predict(self, test_len):
        Y = torch.zeros((self.output_size, test_len), dtype=torch.double)
        u = torch.tensor(self.training_data[-1])  # 使用训练数据的最后一个数据点作为初始输入

        for t in range(test_len):
            # 更新隐藏状态和输出
            self.x = (1 - self.learning_rate) * self.x + self.learning_rate * torch.tanh(
                torch.matmul(self.Win, torch.cat((torch.tensor([1.0]), u))) + torch.matmul(self.W, self.x)
            )
            y = torch.matmul(self.Wout, torch.cat((torch.tensor([1.0]), u, self.x)))
            Y[:, t] = y
            u = y

        return Y
