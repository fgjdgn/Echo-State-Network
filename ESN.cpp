#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic> MatrixXd;

MatrixXd train_and_test_ESN(const MatrixXd& data, int train_len = 2000, int test_len = 100, int init_len = 100, int in_size = 3, int res_size = 1000, double a = 0.7, double reg = 1e-8) {
    // Generate ESN reservoir
    MatrixXd Win = (MatrixXd::Random(res_size, 1 + in_size) - 0.5) * 1;
    MatrixXd W = MatrixXd::Random(res_size, res_size) - 0.5;
    double rhoW = W.eigenvalues().cwiseAbs().real().maxCoeff();
    W *= 1.25 / rhoW;

    // Allocate memory for the reservoir states matrix
    MatrixXd X(1 + in_size + res_size, train_len - init_len);
    MatrixXd Yt = data.block(0, init_len + 1, 1, train_len).transpose();

    MatrixXd x(res_size, 1);
    x.setZero();
    for (int t = 0; t < train_len; t++) {
        MatrixXd u = data.col(t);
        x = (1 - a) * x + a * tanh(Win * MatrixXd::Identity(1 + in_size, 1) * u + W * x);
        if (t >= init_len) {
            X.col(t - init_len) << 1, u, x;
        }
    }

    // Use Ridge Regression to fit the target values
    MatrixXd X_T = X.transpose();
    MatrixXd Wout = (Yt.transpose() * X_T) * (X * X_T + reg * MatrixXd::Identity(1 + in_size + res_size, 1 + in_size + res_size)).inverse();

    // Run the trained ESN in generative mode
    MatrixXd Y(data.rows(), test_len);
    MatrixXd u = data.col(train_len);
    for (int t = 0; t < test_len; t++) {
        x = (1 - a) * x + a * tanh(Win * MatrixXd::Identity(1 + in_size, 1) * u + W * x);
        MatrixXd y = Wout * MatrixXd::Identity(1 + in_size, 1) * u;
        Y.col(t) = y;
        u = y;
    }

    return Y;
}

int main() {
    // Your data here
    MatrixXd data;
    MatrixXd output = train_and_test_ESN(data);

    // Print the output
    std::cout << "Output:\n" << output << std::endl;

    return 0;
}
