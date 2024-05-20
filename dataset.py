class DataSet:
    def __init__(self, array1, array2):
        self.x = array1
        self.y = array2

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def size(self):
        return len(list(zip(self.x, self.y)))