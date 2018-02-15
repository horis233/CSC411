'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    one_digit = 700
    for i in range(10):
        means[i] = np.mean(train_data[i * one_digit:(i + 1) * one_digit], axis=0)

    # Compute means
    return means


def compute_sigma_mles(train_data, means):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    difference = np.zeros((10, 700, 64))
    for i in range(10):
        for j in range(700):
            difference[i][j] = train_data[i][j] - means[i]

    for label in range(10):
        # print ('label: ',label)
        for x in range(64):
            for y in range(64):
                # print ('dimension',x,y)
                for sample in range(train_data.shape[1]):
                    covariances[label][x][y] += difference[label][sample][x] * difference[label][sample][y]

    # Compute covariances: use sum computed last step divided by

    # Compute covariances
    covariances = covariances / 700
    for i in range(10):
        covariances[i] += 0.01 * np.identity(64)
    return covariances


def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_diag = np.zeros((10, 64))
    for i in range(10):
        cov_diag[i] = np.diag(covariances[i])
    for i in range(10):
        plt.subplot(3, 5, i + 1)
        plt.imshow(cov_diag[i].reshape((8, 8)), cmap='gray')
    plt.tight_layout()
    plt.show()


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array 
    '''
    logZ = np.zeros(10)
    for i in range(10):
        # print(cov[i].shape)
        logZ[i] = np.log(np.sqrt(((2 * np.pi) ** 64) * np.linalg.det(covariances[i])))

    likelihood = np.zeros((digits.shape[0], 10))
    for i in range(10):
        x_minus_mean = digits - means[i]
        for n, data in enumerate(x_minus_mean):
            likelihood[n][i] = -(logZ[i] + 0.5 * np.dot(np.dot(data.T, np.linalg.inv(covariances[i])), data))
    return likelihood


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gene_likeihood = generative_likelihood(digits, means, covariances)
    cond = np.sum(np.exp(gene_likeihood - np.log(10)), axis=1).reshape((digits.shape[0],))  # 700 x 1
    cond_standard = np.full((10, digits.shape[0]), cond).T
    con_likelihood = (gene_likeihood - np.log(10)) - np.log(cond_standard)
    return con_likelihood


def avg_conditional_likelihood(digits, labels, means, covariances, stem):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    sum = 0
    for n in range(digits.shape[0]):
        sum += cond_likelihood[n][int(labels[n])]
    AVG = sum/digits.shape[0]
    print('Average conditional likelihood for ' + stem + 'data in correct class is: ', AVG)
    # Compute as described above and return

def leading_eig(cov):
    ld_eig = np.zeros((10, 64))
    for label in range(10):
        w, v = np.linalg.eig(cov[label])
        ld_eig[label] = v[np.argmax(w)]
    for i in range(10):
        plt.subplot(3, 5, i + 1)
        plt.imshow(ld_eig[i].reshape((8, 8)), cmap='gray')
    plt.tight_layout()
    plt.show()


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def accuracy(labels, digits, means, covariance):
    '''
    acc = np.zeros(10)
    for i in range(10):
        acc[i] = labels[i] == classify_data(digits[i], means[i], covariance[i])
    return acc.mean()
    '''
    return np.equal(labels, classify_data(digits, means, covariance)).mean()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data', shuffle=False)
    labels = np.arange(10)
    # Fit the model
    means = compute_mean_mles(train_data)
    covariances = compute_sigma_mles(train_data.reshape((10, -1, 64)), means)
    plot_cov_diagonal(covariances)
    leading_eig(covariances)
    print('Train_data: ')
    avg_conditional_likelihood(train_data, train_labels, means, covariances, data.TRAIN_STEM)
    print('Test_data: ')
    avg_conditional_likelihood(test_data, test_labels, means, covariances, data.TEST_STEM)
    print('\nThe accuracy of train data is: ', accuracy(train_labels, train_data, means, covariances))
    print('The accuracy of test data is: ', accuracy(test_labels, test_data, means, covariances))

    # Evaluation


if __name__ == '__main__':
    main()
