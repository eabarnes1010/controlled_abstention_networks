"""Classification with abstention gameboard-type synthetic data."""

__author__ = "Elizabeth A. Barnes and Randal J. Barnes"
__date__ = "January 11, 2021"

import numpy as np


class Gameboard:
    """Gameboard-type synthetic data for abstention experiments.

    Attributes
    ----------
    nrow : int
        The number of rows of pixels in the gameboard.

    ncol : int
        The number of columns of pixels in the gameboard.

    nrank : int
        The number of rows of cells in the gameboard.

    nfile : int
        The number of columns of cells in the gameboard.

    nlabel : int
        The number of unique labels in the gameboard.

    ntranquil : int
        The number of tranquil cells in the gameboard.

    pr_validation : float
        The probability that a random pixel will be assigned for validation.

    pr_mislabel : float
        The probability that a random pixel in a noisy cell will be mislabeled.

    Methods
    -------
    getlabel(row, col)
        Returns the label of the pixel at (row, col).

    istranquil(row, col)
        Returns true if the pixel at (row, col) is in a tranquil cell.

    generate()
        Returns generated data partitioned into training and validation subsets.

    """
    def __init__(self, nrow, ncol, nrank, nfile, nlabel, ntranquil, pr_validation, pr_mislabel):
        
        self.nrow = nrow
        self.ncol = ncol
        self.nrank = nrank
        self.nfile = nfile
        self.nlabel = nlabel
        self.ntranquil = ntranquil
        self.pr_validation = pr_validation
        self.pr_mislabel = pr_mislabel

        self.ncell = self.nrank * self.nfile
        self.nnoisy = self.ncell - self.ntranquil

        self.board = np.reshape(
            np.random.permutation(
                np.array(range(self.ncell), dtype=int) % self.nlabel
            ),
            (self.nrank, self.nfile)
        )

        self.tranquil = np.reshape(
            np.random.permutation(
                np.append([True] * self.ntranquil, [False] * self.nnoisy)
            ),
            (self.nrank, self.nfile)
        )

    def __repr__(self):
        return (
            f"Gameboard({self.nrow}, {self.ncol}, {self.nrank}, "
            + f"{self.nfile}, {self.nlabel}, {self.ntranquil}, "
            + f"{self.pr_validation}, {self.pr_mislabel})"
        )

    def __str__(self):
        return (
                "\n"
                + f"nrow = {self.nrow}\n"
                + f"ncol = {self.ncol}\n"
                + f"nrank = {self.nrank}\n"
                + f"nfile = {self.nfile}\n"
                + f"nlabel = {self.nlabel}\n"
                + f"ntranquil = {self.ntranquil}\n"
                + f"pr_validation = {self.pr_validation}\n"
                + f"pr_mislabel = {self.pr_mislabel}\n"
                + f"board = \n{self.board}\n"
                + f"tranquil = \n{self.tranquil}"
        )

    def getlabel(self, row, col):
        """Returns the label the pixel at (row, col)."""
        rank = int(row * self.nrank / self.nrow)
        file = int(col * self.nfile / self.ncol)
        return self.board[rank, file]

    def getlabels(self, X):
        """Returns an ndarray of labels for the pixels at X = (col, row)."""
        label = []
        for col, row in X:
            rank = int(row * self.nrank / self.nrow)
            file = int(col * self.nfile / self.ncol)
            label.append(self.board[rank, file])
        return np.array(label, dtype=int)

    def istranquil(self, row, col):
        """Returns true if the pixel at (row, col) is in a tranquil cell."""
        rank = int(row * self.nrank / self.nrow)
        file = int(col * self.nfile / self.ncol)
        return self.tranquil[rank, file]

    def aretranquil(self, X):
        """Returns a boolean array indicating which of the pixels at
        X = (col, row) are in tranquil cells."""
        tranquility = []
        for col, row in X:
            rank = int(row * self.nrank / self.nrow)
            file = int(col * self.nfile / self.ncol)
            tranquility.append(self.tranquil[rank, file])
        return np.array(tranquility, dtype=bool)

    def generate(self):
        """Returns generated data partitioned into training and validation subsets."""
        X_all = []
        y_all = []

        X_train = []
        y_train = []

        X_val = []
        y_val = []

        for col in range(self.ncol):
            for row in range(self.nrow):
                label = self.getlabel(row, col)
                if not self.istranquil(row, col) and np.random.random() < self.pr_mislabel:
                    label = (label + np.random.randint(low=1, high=self.nlabel)) % self.nlabel

                X_all.append([col, row])
                y_all.append(label)

                if np.random.random() > self.pr_validation:
                    X_train.append([col, row])
                    y_train.append(label)
                else:
                    X_val.append([col, row])
                    y_val.append(label)

        X_all = np.array(X_all, dtype=int)
        y_all = np.array(y_all, dtype=int)

        X_train = np.array(X_train, dtype=int)
        y_train = np.array(y_train, dtype=int)

        X_val = np.array(X_val, dtype=int)
        y_val = np.array(y_val, dtype=int)

        return X_all, y_all, X_train, y_train, X_val, y_val
