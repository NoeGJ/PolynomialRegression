from polynomial_regression import PolynomialRegression

class Main:
    def __init__(self):
        self.x = [108,115,106,97,95,91,97,83,83,78,54,67,56,53,61,115,81,78,30,45,99,32,25,28,90,89]
        self.y = [95,96,95, 97, 93,94,95,93,92,86,73,80,65,69,77,96,87,89,60,63,95,61,55,56,94,93]
        self.x2 = [self.x[i]**2 for i in range(len(self.x))]
        self.x3 = [self.x[i]**3 for i in range(len(self.x))]

        #self.xpredict = [93, 95, 101]

        self.xpredict = [28, 90,89, 27,102, 80]

    def printLinear(self):
        X = [[i] for i in self.x]
        print("Linear")
        dm = PolynomialRegression(X, self.y)
        for i in range(len(dm.getBetas())):
            print("B{} = ".format(i), dm.getBetas()[i])
        dm.toprintEq(self.xpredict)
        print("R^2 = ", dm.getRSquared())
        print("R = ",dm.getR())

    def printQuadratic(self):
        X = [[i, j] for i, j in zip(self.x, self.x2)]
        #print(X)
        print("Quadratic")
        dm = PolynomialRegression(X, self.y)
        for i in range(len(dm.getBetas())):
            print("B{} = ".format(i), dm.getBetas()[i])
        dm.toprintEq(self.xpredict)
        print("R^2 = ", dm.getRSquared())
        print("R = ",dm.getR())

    def printCubic(self):
        X = [[i, j, k] for i, j, k in zip(self.x, self.x2, self.x3)]
        #print(X)
        print("Cubic")
        dm = PolynomialRegression(X, self.y)
        for i in range(len(dm.getBetas())):
            print("B{} = ".format(i), dm.getBetas()[i])
        dm.toprintEq(self.xpredict)
        print("R^2 = ", dm.getRSquared())
        print("R = ",dm.getR())


if __name__ == '__main__':
    ho = Main()
    ho.printLinear()
    ho.printQuadratic()
    ho.printCubic()