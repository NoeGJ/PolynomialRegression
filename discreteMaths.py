import math
import numpy as np
import dataset

class DiscreteMaths(dataset.DataSet):
    def __init__(self, array1, array2):
        super().__init__(array1, array2)

    def __copyAndInitMatrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        return [[0 for x in range(rows)] for x in range(cols)]

    def mean(self, array):
        return sum(array) / len(array)

    def transposeMatrix(self, matrix):
        newarr = self.__copyAndInitMatrix(matrix)

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                newarr[j][i] = matrix[i][j]

        return newarr

    def multiMatrix(self, matrix1, matrix2):
        newarr = [[0 for x in range(len(matrix2[0]))] for y in range(len(matrix1)) ]

        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    newarr[i][j] += matrix1[i][k] * matrix2[k][j]
        return newarr

    def matrixInverse(self, matrix):
        return np.linalg.inv(matrix)