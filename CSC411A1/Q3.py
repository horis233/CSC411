import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sympy import *

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''

    Loss = np.zeros(X.shape)
    for i in range(len(X)):
        Loss[i] = 2*X[i].T*np.dot(X[i].T, w)-2*X[i].T*y[i]
    gradient = np.mean(Loss, axis=0)

    return gradient

    #raise NotImplementedError()

def var_grad_m(batch_sampler,m,w,index):
    sum_grad = 0
    for n in range(500):
        X_b, y_b = batch_sampler.get_batch(m)
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        var_grad = np.var(batch_grad[index])
        sum_grad += var_grad

    return sum_grad/500



def main():

    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)

    # Example usage
    num_iter = 500

    # Question 5
    batch_grad = []
    for j in range (num_iter):
        X_b, y_b = batch_sampler.get_batch(m=50)
        batch_grad.append(lin_reg_gradient(X_b, y_b, w))
    grad_batch = np.mean(batch_grad, axis=0)
    grad_true = lin_reg_gradient(X, y, w)
    cosine_s = cosine_similarity(grad_batch, grad_true)
    print('cosine similarity is ' + str(cosine_s))
    distance = ((grad_batch-grad_true)**2).mean()
    distance = sqrt(distance)
    print('squared distance metric is ' + str(distance))

    #Question 6

    index = random.randint(0, 12)
    E_gradient = grad_true[index]
    y_m=[]
    x_m=[]
    for m in range(1, 401):

        var_grad = []
        for n in range(500):
            X_b, y_b = batch_sampler.get_batch(m)
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            var_grad.append(( batch_grad[index] - E_gradient)**2)
        y_m.append(log(np.mean(var_grad)))
        x_m.append(log(m))

    plt.plot(x_m, y_m)
    plt.xlabel("log m")
    plt.ylabel("log variance")
    plt.show()




if __name__ == '__main__':
    main()
