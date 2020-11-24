import numpy as np
import matplotlib.pyplot as plt

class DataSetGenerator:
    """
    A class for generating a data set of a specified size with 10 features
    and 4 classes of objects
    """
    
    #Some lable names
    labelNames = ['Ель', 'Пихта', 'Сосна', 'Кедр']
    featureNames = [
            'Средняя длина иглы (см)',
            'Глубина залегания корня(м)',
            'Среднее число семян в шишке',
            'Число шишек',
            'Максимальная длина ветви',
            'Высота дерева',
            'Иглы являются ломкими',
            'Среднее расстояние между соседними иглами(мм)',
            'Предел прочности при сжатии вдоль волокон(МПа)',
            'Предел прочности при сжатии поперек волокон(МПа)'
    ]

    labelBounds = [[] for i in range(4)]
    labelBounds[0] = [
            [0.5, 2],
            [5, 8],
            [100, 150],
            [90, 150],
            [3, 4],
            [3, 6],
            [0, 0.5],
            [2, 4],
            [50, 60],
            [8, 11]
    ]
    labelBounds[1] = [
            [0.3, 1.5],
            [2, 6],
            [120, 160],
            [50, 100],
            [1, 3],
            [1.5, 4],
            [0, 0.6],
            [0.5, 1.5],
            [45, 55],
            [9, 12]
    ]
    labelBounds[2] = [
            [1.5, 4],
            [6, 10],
            [80, 130],
            [70, 130],
            [3, 5],
            [2, 5],
            [0.3, 0.7],
            [3, 6],
            [70, 85],
            [12, 15]
    ]
    labelBounds[3] = [
            [2, 4.5],
            [4, 7],
            [140, 180],
            [70, 100],
            [1, 4],
            [1.9, 6],
            [0.6, 1],
            [2, 4],
            [60, 76],
            [10, 14]
    ]


    def generateDataSet(self, sizeForOneClass, seed=1):
        """
        Generate a data set
    
        Parameters
        ----------
        sizeForOneClass : an integer number
            number of objects of each of four classes 
        
        Returns
        ----------
        A dictionary:
            X - array of data of shape (4*sizeForOneClass, 10)
            Y - numpy array of lables of size 4*sizeForOneClass 
        """

        labelBounds = self.labelBounds
        np.random.seed(seed)
        dataSet = []
        for lable in range(4):
            for k in range(sizeForOneClass):
                newObject = {'x': [np.random.rand()*(labelBounds[lable][i][1]-labelBounds[lable][i][0]) + labelBounds[lable][i][0] for i in range(10)], 'y': lable}
                dataSet.append(newObject)
                
        dataSet = np.array(dataSet)
        np.random.shuffle(dataSet)
        
        X = np.array([dataSet[i]['x'] for i in range(len(dataSet))])
        Y = np.array([dataSet[i]['y'] for i in range(len(dataSet))])
        return {'X': X, 'Y': Y}


    def normalizeX(self, X):
        """
        Parameters
        ----------
        X : numpy array
            a data set
        
        Returns
        ----------
        X : numpy array
            modified data set
        """
        for colInd in range(X.shape[1]):
            X[:, colInd] = (X[:, colInd] - np.min(X[:, colInd]))/(np.max(X[:, colInd]) - np.min(X[:, colInd]))
        
        return X
    
    
    def drawDataSet(self, X, Y, featureInd1, featureInd2):
        """
        Draw a 2d picture of data set
        
        Parameters
        ----------
        X : numpy array
            training data set
        Y : numpy array
            lables
        featureInd1 : number from 0 to 9
            number of feature for horizontal axis
        featureInd2 : number from 0 to 9
            number of feature for vertical axis
        """
        colorMap = ['r.', 'g.', 'b.', 'y.']
        for i in range(Y.shape[0]):
            plt.plot(X[i][featureInd1], X[i][featureInd2], colorMap[Y[i]])
        plt.xlabel(self.featureNames[featureInd1])
        plt.ylabel(self.featureNames[featureInd2])
        plt.show()
        