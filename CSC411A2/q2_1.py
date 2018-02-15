'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''
from sklearn.model_selection import KFold

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data ** 2).sum(axis=1).reshape(-1, 1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        train_norm = (self.train_data ** 2).sum(axis=1).reshape(-1, 1)
        test_norm = (test_point ** 2).sum(axis=1).reshape(1, -1)
        dist = self.train_norm + test_norm - 2 * self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''
        l2 = self.l2_distance(test_point)
        sorted_l2 = np.argsort(l2)
        classCount = {}
        for i in range(k):
            voteLabel = self.train_labels[sorted_l2[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        maxCount = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        return maxIndex


def cross_validation(knn, k_range=np.arange(1, 16)):
    optimal_k = 0
    optimal_accuracy = 0
    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        accuracies = k_fold(knn.train_data, knn.train_labels, k)
        print('K for KNN and the corresponding mean k_fold accuracy: ', k, '&', accuracies.mean())
        if accuracies.mean() >= optimal_accuracy:
            optimal_accuracy = accuracies.mean()
            optimal_k = k

    return optimal_k, optimal_accuracy


def k_fold(data, label, k):
    i = 0
    Kf = KFold(10)
    accuracies = np.zeros(10)
    for index_train, index_valid in Kf.split(data, label):
        train_x, valid_x = data[index_train], data[index_valid]
        train_y, valid_y = label[index_train], label[index_valid]
        knn = KNearestNeighbor(train_x, train_y)
        accuracies[i] = classification_accuracy(knn, k, valid_x, valid_y)

        i = i + 1
    return accuracies


def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    res = np.zeros(eval_data.shape[0])
    for i in range(eval_data.shape[0]):
        res[i] = knn.query_knn(eval_data[i], k) == eval_labels[i]
    return res.mean()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    print('Accuracy for train data with k=1: ',classification_accuracy(knn, 1, train_data, train_labels))
    print('Accuracy for train data with k=15: ',classification_accuracy(knn, 15, train_data, train_labels))
    print('Accuracy for test data with k=1: ',classification_accuracy(knn, 1, test_data, test_labels))
    print('Accuracy for test data with k=15: ',classification_accuracy(knn, 15, test_data, test_labels))

    k, loss = cross_validation(knn)
    print('\nOptimal K for KNN and the corresponding mean k_fold accuracy: ', k, '&', loss)
    print('Accuracy for train data with optimal k: ', classification_accuracy(knn, k, train_data, train_labels))
    print('Accuracy for test data with optimal k: ', classification_accuracy(knn, k, test_data, test_labels))


if __name__ == '__main__':
    main()
