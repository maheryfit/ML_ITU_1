import numpy as np

class MyLinearRegression:
    def __init__(self):
        self.theta_n = None
        self.theta_0 = None

    def fit(self, x_var, y_real):
        # Add bias term (column of 1s) to X
        #print(x_var)
        X_b = np.c_[np.ones((x_var.shape[0], 1)), x_var]  # Add x0 = 1 to each instance
        # Normal Equation: theta = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_real)
        self.theta_0 = theta[0]
        self.theta_n = theta[1:]

    def predict(self, x_var):
        # Add bias term (column of 1s) to X
        print(x_var.to_string())
        X_b = np.c_[np.ones((x_var.shape[0], 1)), x_var] # Add x0 = 1 to each instance
        # np.r_ (Merge the first Vector [0] with the Vector [1])
        return X_b.dot(np.r_[self.theta_0, self.theta_n])

    def get_r2_score(self, x_var, y_real):
        """
        Formule: 1 - [sum((y[i] - y_predicted[i]) ^ 2) - sum((y[i] - y_mean) ^ 2)]
        """
        y_predicted = self.predict(x_var)
        ss_reg = np.sum((y_real - y_predicted) ** 2)
        ss_tot = np.sum((y_real - float(np.mean(y_real))) ** 2)
        return 1 - (ss_reg / ss_tot)

    def get_rmse(self, x_var, y_real):
        """
        Formule: np.sqrt((1 / n) * sum((y[i] - y_predicted[i]) ^ 2))
        """
        y_predicted = self.predict(x_var)
        return np.sqrt(np.sum((y_real - y_predicted) ** 2) / y_real.shape[0])
