train_and_test_ESN <- function(data, train_len = 2000, test_len = 100, init_len = 100, 
                              in_size = 3, res_size = 1000, a = 0.7, reg = 1e-8) {
  # Generate ESN reservoir
  set.seed(42)
  Win <- matrix(runif(res_size * (1 + in_size), -0.5, 0.5), nrow = res_size)
  W <- matrix(runif(res_size * res_size, -0.5, 0.5), nrow = res_size)
  rhoW <- max(Mod(eigen(W)$values))
  W <- W * (1.25 / rhoW)

  # Allocate memory for the reservoir states matrix
  X <- matrix(0, nrow = 1 + in_size + res_size, ncol = train_len - init_len)
  Yt <- data[(init_len + 1):(train_len + 1)]

  x <- matrix(0, nrow = res_size, ncol = 1)
  for (t in 1:train_len) {
    u <- data[t]
    x <- (1 - a) * x + a * tanh(Win %*% c(1, u) + W %*% x)
    if (t >= init_len) {
      X[, t - init_len] <- c(1, u, x)
    }
  }

  # Use Ridge Regression to fit the target values
  X_T <- t(X)
  Wout <- t(data[(init_len + 1):(train_len + 1)]) %*% X_T %*% solve(X %*% X_T + reg * diag(1 + in_size + res_size))

  # Run the trained ESN in generative mode
  Y <- matrix(0, nrow = length(data), ncol = test_len)
  u <- data[train_len]
  for (t in 1:test_len) {
    x <- (1 - a) * x + a * tanh(Win %*% c(1, u) + W %*% x)
    y <- Wout %*% c(1, u, x)
    Y[, t] <- y
    u <- y
  }

  return(Y)
}

# Usage example:
data <- # Your data here
output <- train_and_test_ESN(data, train_len = 2000, test_len = 100, init_len = 100, 
                             in_size = 3, res_size = 1000, a = 0.7, reg = 1e-8)
