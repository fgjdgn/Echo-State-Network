function output = train_and_test_ESN(data, train_len, test_len, init_len, in_size, res_size, a, reg)
    % Generate ESN reservoir
    rng(42);
    Win = (rand(res_size, 1 + in_size) - 0.5) * 1;
    W = rand(res_size, res_size) - 0.5;
    rhoW = max(abs(eig(W)));
    W = W * (1.25 / rhoW);

    % Allocate memory for the reservoir states matrix
    X = zeros(1 + in_size + res_size, train_len - init_len);
    Yt = data(init_len + 1:train_len + 1);

    x = zeros(res_size, 1);
    for t = 1:train_len
        u = data(t);
        x = (1 - a) * x + a * tanh(Win * [1; u] + W * x);
        if t >= init_len
            X(:, t - init_len) = [1; u; x];
        end
    end

    % Use Ridge Regression to fit the target values
    X_T = X.';
    Wout = (data(init_len + 1:train_len + 1).' * X_T) * inv(X * X_T + reg * eye(1 + in_size + res_size));

    % Run the trained ESN in generative mode
    Y = zeros(size(data, 1), test_len);
    u = data(train_len);
    for t = 1:test_len
        x = (1 - a) * x + a * tanh(Win * [1; u] + W * x);
        y = Wout * [1; u; x];
        Y(:, t) = y;
        u = y;
    end

    output = Y;
end

% Usage example:
data = % Your data here
output = train_and_test_ESN(data, 2000, 100, 100, 3, 1000, 0.7, 1e-8);
