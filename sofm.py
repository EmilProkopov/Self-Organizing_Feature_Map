import numpy as np
import matplotlib.pyplot as plt

class SOFM:
    """
    Realization of self-organizing feature map
    
    params : dictionary of model parameters
    """
    params = {}


    def initParams(self, hiddenLayerSize, spaceSize):
        """
        Randomly initialize model parameters
        
        Parameters
        ----------
        hiddenLayerSize : integer number
            number of nodes in the hidden layer
        spaceSize: integer number
            number of features in each object in the data set

        """
        c = np.random.rand(hiddenLayerSize, spaceSize)
        denom = np.random.rand(hiddenLayerSize)
        W = np.random.randn(spaceSize, hiddenLayerSize) * 0.01
        self.params = {'c': c, 'd': denom, 'W': W}


    def applyKMeans(self,
                    X,
                    numIterations = 1000,
                    learningRate = 0.01,
                    showProgress=False):
        """
        Use KMeans algorithm to change centers of SOFM after initialization 
        
        Parameters
        ----------
        X : numpy array
            a data set
        numIterations : number
            number of iterations
        learningRate : number
            learning rate
        showProgress: boolean
            to print something each 10000 iterations or not
        """
        x_ind = 0
        for i in range(numIterations):
            if showProgress and (i+1)%10000 == 0:
                print(str(i) + ' / ' + str(numIterations))
            
            if x_ind == len(X):
                x_ind = 0

            x = X[x_ind]
            winnerInd = np.argmin(np.linalg.norm(self.params['c']-x, axis=1))
            tmp = self.params['c'][winnerInd]
            self.params['c'][winnerInd] = tmp + learningRate * (x - tmp)
            
            x_ind += 1

        tmp2 = self.params['c']
        for i in range(len(self.params['d'])):
            closest = sorted(tmp2,
                             key=(lambda val: np.sum((val-tmp2[i])**2)))[:3]
            self.params['d'][i] = np.sqrt(
                    np.sum((closest-tmp2[i])**2)/len(closest))


    def drawNodeCenters(self, featureInd1, featureInd2):
        """
        Draw a 2d picture of node coordinates
        
        Parameters
        ----------
        featureInd1 : integer number
            number of feature for horizontal axis
        featureInd2 : integer number
            number of feature for vertical axis
        """
        for i in range(self.params['c'].shape[0]):
            plt.plot(self.params['c'][i][featureInd1],
                     self.params['c'][i][featureInd2], 'ko')
        plt.xlabel(str(featureInd1))
        plt.ylabel(str(featureInd2))
        plt.show()


    def _activationRadialGauss(self, x):
        """
        Calculate Gauss activations of nodes 
        
        Parameters
        ----------
        x : numpy array
            an object from data set
        
        Returns
        ----------
        numpy array of activations
        """
        tmp1 = self.params['d']
        tmp = [np.exp(-np.sum((x-self.params['c'][i])**2)/
                      (2*(tmp1[i]))) for i in range(len(tmp1))]

        return np.array(tmp)
    
    
    def _activationLinear(self, a):
        """
        Parameters
        ----------
        a : numpy array
            return value of _activationRadialGauss
        
        Returns
        ----------
        numpy array of linear activations
        """
        return np.dot(self.params['W'], a)
    
    
    def _activationFarward(self, x):
        """
        Combination of _activationLinear and _activationRadialGauss
        """
        return self._activationLinear(self._activationRadialGauss(x))
    
    
    def train(self,
              X,
              mode='WTA',
              numIterations = 1000,
              learningRate = 0.01,
              showProgress = False):
        """
        Train the model

        Parameters
        ----------
        X : numpy array
            a data set
        
        mode : string
            learning algorythm: 'WTA' or 'Kahonin'
        
        numIterations : number
            number of iterations
        
        learningRate : number
            learning rate
        
        showProgress : boolean
            to print something each 10000 iterations or not
        """
        S = (lambda ind, w: 1 if ind in w else 0)
        #S = (lambda ind, w: 1)
        for i in range(numIterations):
            if showProgress and (i+1)%10000 == 0:
                print(str(i) + ' / ' + str(numIterations))
                
            x = X[i%len(X)]
            a = self._activationFarward(x)

            winnerInd = [np.argmax(a)] if mode == 'WTA' else (-a).argsort()[:3]
            for j in range(self.params['W'].shape[1]):
                self.params['W'][:,j] = self.params['W'][:,j] + \
                learningRate * S(j, winnerInd) * (x - self.params['W'][:,j])
                
    
    def apply(self, X):
        """
        Apply trained params to a data set
        
        Parameters
        ----------
        X : numpy array
            a data set
        
        Returns
        ----------
        A new data set
        """
        return np.array([self._activationFarward(x) for x in X])
        