#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Classes for metagenomics analysis
'''
import numpy as np

class ModifiedMetrics:

    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.target - self.predict(self.data)) ** 2
        self.sq_error_ = np.sum(squared_errors)
        return self.sq_error_

    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_y = np.mean(self.target)
        squared_errors = (self.target - avg_y) ** 2
        self.sst_ = np.sum(squared_errors)
        return self.sst_

    def r_squared(self):
        '''returns calculated value of r^2'''
        self.r_sq_ = 1 - self.sse()/self.sst()
        return self.r_sq_

    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2'''
        self.adj_r_sq_ = 1 - (self.sse()/self._dfe) / (self.sst()/self._dft)
        return self.adj_r_sq_

    def mse(self):
        '''returns calculated value of mse'''
        self.mse_ = np.mean( (self.predict(self.data) - self.target) ** 2 )
        return self.mse_

    def pretty_print_stats(self):
        '''returns report of statistics for a given model object'''
        items = ( ('sse:', self.sse()), ('sst:', self.sst()),
                 ('mse:', self.mse()), ('r^2:', self.r_squared()),
                  ('adj_r^2:', self.adj_r_squared()))
        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))


class MyLinearRegressionWithInheritance(ModifiedMetrics):

    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self._fit_intercept = fit_intercept

    def __repr__(self):
        return None

    def fit(self, X, y):
        """
        Fit model coefficients.

        Arguments:
        X: 1D or 2D numpy array
        y: 1D numpy array
        """

        # training data & ground truth data
        self.data = X
        self.target = y

        # degrees of freedom population dep. variable variance
        self._dft = X.shape[0] - 1
        # degrees of freedom population error variance
        self._dfe = X.shape[0] - X.shape[1] - 1

        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)

        # add bias if fit_intercept
        if self._fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # closed form solution
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self._fit_intercept:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

    def predict(self, X):
        """Output model prediction.

        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        return self.intercept_ + np.dot(X, self.coef_)


class model:
    def __init__(self):
        self.inp, self.mem, self.pre, self.act = 0, 0, 0, 0
        self.votearray = []

    def __repr__(self):
        return f"inp:{bin(self.inp)[:1:-1]}\nmem:{bin(self.mem)[:1:-1]}\npre:{bin(self.pre)[:1:-1]}\nact:{bin(self.act)[:1:-1]}"

    def io(self, files={}, mode='in'):
        self.files = dict()
        def parse(obj=0, mode='in', filename=''):
            if mode == "in":
                if not os.path.isfile(filename):
                    with open(filename, "wb") as of:
                        of.write(b'\0')
                with open(filename, "rb") as of:
                    return int.from_bytes(of.read(), byteorder="big")
            elif mode == "out":
                with open(filename, "wb") as of:
                    of.write(obj.to_bytes((obj.bit_length() + 7) // 8, 'big') or b'\0')

        self.files = dict(files)
        if not self.files:
            self.files["inp"] = "data/input"
            self.files["mem"] = "data/memory"
            self.files["pre"] = "data/predict"
        self.inp = parse(obj=self.inp, mode=mode, filename=self.files["inp"])
        self.mem = parse(obj=self.mem, mode=mode, filename=self.files["mem"])
        self.pre = parse(obj=self.pre, mode=mode, filename=self.files["pre"])
        return self

    def ols(X, y):
        '''returns parameters based on Ordinary Least Squares.'''
        xtx = np.dot(X.T, X) ## x-transpose times x
        inv_xtx = np.linalg.inv(xtx) ## inverse of x-transpose times x
        xty = np.dot(X.T, y) ## x-transpose times y
        return np.dot(inv_xtx, xty)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
