import numpy as np
from numpy.typing import NDArray


class LinearRegression:

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 100_000, tol:float = 1e-6) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol = tol

    
    def _get_gradient(self, X: NDArray, y: NDArray, w: NDArray, b:float) -> tuple[NDArray, float, float]:

        m = X.shape[0]

        err = np.matmul(X, w) + b - y
        dj_dw = np.matmul(X.T, err) / m
        dj_db = err.mean()
        cost = (err ** 2).mean() / 2

        return dj_dw, dj_db, cost
        

    def _check_params(self, X: NDArray, y: NDArray) -> bool:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be array")

        if X.ndim != 2:
            raise ValueError("X must be 2D array")

        if y.ndim != 1:
            raise ValueError("y must be 1D array")

        if X.shape[0] != y.shape[0]:
             raise ValueError("X and y must have the same number of samples")


    def _check_X(self, X: NDArray) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a NumPy array")
    
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
    
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has an incorrect number of features")

    def fit(self, X_train: NDArray, y_train: NDArray) -> "LinearRegression":
        self._check_params(X_train, y_train)

        self.coef_ = np.zeros(X_train.shape[1])
        self.intercept_ = 0.0

        self.loss_hist = []
        
        for i in range(1, self.n_iterations):          
            
            grad_w, grad_b, cost = self._get_gradient(X_train, y_train, self.coef_, self.intercept_)
            self.coef_ = self.coef_ - self.learning_rate * grad_w
            self.intercept_ = self.intercept_ - self.learning_rate * grad_b

            self.loss_hist[i] = cost

            if len(seelf.loss_hist) > 0 and abs(self.loss_hist[-2] - cost) < self.converge:
                break

        return self
            
        

    def predict(self, X_test: NDArray) -> NDArray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("This model is not fitted yet. Call fit() first.")
        
        self._check_X(X_test)

        return np.matmul(X_test, self.coef_) + self.intercept_
        
    def score(self, X: NDArray, y: NDArray) -> float:
        self._check_params(X, y)

        y_pred = self.predict(X)
    
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
    
        return 1 - ss_res / ss_tot
    