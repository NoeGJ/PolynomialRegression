import math
from copy import deepcopy

import discreteMaths

class PolynomialRegression( discreteMaths.DiscreteMaths ):
    def __init__(self, array1, array2 ):
        super().__init__(array1, array2)
        self.ymatrix = [[array2[x]] for x in range(len(array2))]

        new = deepcopy(self.x)
        for i in range(len(new)):
            new[i].insert(0, 1)

        self.beta = self.__toComputeBeta(new, self.ymatrix)

        self.rq = self.__toComputeRSquared(self.x)
        self.r = self.__toComputeR()


    def __toComputeBeta(self, X, y): # (X^t * X)^-1 * X^t * y
        Xt = self.transposeMatrix(X)
        XtX = self.multiMatrix(Xt, X)
        Xty = self.multiMatrix(Xt, y)
        XX_1 = self.matrixInverse(XtX)
        return self.multiMatrix(XX_1, Xty)

    def getBeta0(self):
        return self.beta[0][0]

    def getBetas(self):
        return self.beta

    def getR(self):
        return self.r

    def getRSquared(self):
        return self.rq

    def toPredict(self, x_n): # x_n es una lista
        result = self.getBeta0() # inicializa el acumulador result al valor de beta0

        for i in range(len(x_n)): # recorre la lista y el valor los multiplica por los beta
                                    # la lista "x_n" tiene que tener la misma cantidad de valores que la lista de betas
            result += self.beta[i + 1][0] * x_n[i] # Bx^1 + Bx^2 + Bx^n
        return result

    def __toComputeRSquared(self, array): # array = [ x^1, x^2, x^n ]
        ssr = sum((yi - (self.toPredict(xi))) ** 2 for xi, yi in zip(array, self.y))
        tss = sum((yi - self.mean(self.y)) ** 2 for yi in self.y)
        return 1 - (ssr / tss)

    def __toComputeR(self):
        return math.sqrt(self.rq)

    def toprintEq(self, values):
        for value in values:
            arr = list()
            print("y^ = {}".format(self.getBeta0()), end=" ")
            for i in range(len(self.beta) - 1):
                arr.append(value ** (i + 1))
                print(" + {} * {}".format(self.beta[i + 1][0], value ** (i + 1)), end=" ")

            print("= {}".format(self.toPredict(arr)))
