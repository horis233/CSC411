import numpy as np

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)


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


class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, beta=0.0):
        self.lr = lr
        self.beta = beta

    def update_params(self, params, grad, data):

        data = self.beta*data - self.lr*grad

        params += data

        # Update parameters using GD with momentum and return
        # the updated parameters
        return params, data


class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        self.w_r = self.w
        self.w_r[0] = 0
        self.w_w = self.w
        self.w_w[0] = 1

    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''

        loss = []
        for i, x in enumerate(X):
            loss.append(np.where((y[i] * np.dot(np.transpose(self.w), x)) >= 1, 0, 1-y[i] * np.dot(np.transpose(self.w), x)))
        loss_m = np.array(loss).mean()
        return loss_m

    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        hinge = np.where((y * (np.dot(X, self.w_w.reshape(-1, 1)))) >= 1, 0, -y * X).mean(axis=0)
        grad = self.w_r + self.c * hinge
        return grad

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1

        return np.array([1 if np.dot(np.transpose(self.w), x) >= 0 else -1 for x in X])
        #return np.where(np.dot(X, self.w.reshape((-1, 1))) > 0, 1, -1)


def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]

    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets


def optimize_test_function(optimizer, w_init=10.0, steps=200):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''

    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    vel = 0
    for _ in range(steps):
        w, vel=optimizer.update_params(w, func_grad(w), vel)
        w_history.append(w)
        # Optimize and update the history
        pass
    return w_history


def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    batch_sampler = BatchSampler(train_data, train_targets, batchsize)
    svm = SVM(penalty, train_data.shape[1])
    data = np.zeros_like(svm.w)
    for _ in range(iters):
        x_batch, y_batch = batch_sampler.get_batch()
        svm.w,data = optimizer.update_params(svm.w, svm.grad(x_batch, y_batch),data)
    return svm


def calculate(beta):

    optimizer = GDOptimizer(0.05, beta)
    svm = optimize_svm(train_data, train_targets.reshape((-1, 1)), 1.0, optimizer, 100, 500)
    train_pred = svm.classify(train_data)
    train_accuracy = np.equal(train_targets, train_pred).mean()
    test_pred = svm.classify(test_data)
    test_accuracy = np.equal(test_targets, test_pred).mean()
    print("Beta == {}" .format(beta))
    print('The train loss of model with beta = {} : {}'.format(beta, str(svm.hinge_loss(train_data, train_targets.reshape((-1, 1))))))
    print('The test loss of model with beta = {} : {}'.format(beta, str(svm.hinge_loss(test_data, test_targets.reshape((-1, 1))))))
    print('The train accuracy of model with beta = {} : {}'.format(beta, train_accuracy))
    print('The test accuracy of model with beta = {} : {}\n\n'.format(beta, test_accuracy))

    plt.imshow(np.reshape(svm.w[1:], (-1, 28)), cmap='gray')
    plt.show()

if __name__ == '__main__':

    gd_zero = GDOptimizer(1.0)
    w_zero = optimize_test_function(gd_zero)
    gd_pointNine = GDOptimizer(1.0, 0.9)
    w_pointNine = optimize_test_function(gd_pointNine)

    plt.plot(w_zero, '.')
    plt.plot(w_pointNine, '+')
    plt.xlabel('iterations')
    plt.ylabel('w_val')
    plt.show()

    train_data, train_targets, test_data, test_targets = load_data()
    train_data = np.concatenate((np.ones((train_data.shape[0], 1)), train_data), axis=1)
    test_data = np.concatenate((np.ones((test_data.shape[0], 1)), test_data), axis=1)

    calculate(0.0)
    calculate(0.1)
