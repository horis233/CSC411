'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)


def compute_parameters(train_data):
    '''
    Compute the eta MAP estimate/MLE with augmented data

    You should return a numpy array of shape (10, 64)
    where the ith row corresponds to the ith digit class.
    '''
    eta = (train_data.sum(axis=1) + 1) / (train_data.shape[1] + 2)
    return eta


def plot_images(X):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i].reshape((8, 8)), cmap='gray')

    # Plot all means on same axis
    plt.tight_layout()
    plt.show()


def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''

    plot_images(np.array([[np.random.binomial(1, eta[i, j]) for j in range(64)] for i in range(10)]))


def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array 
    '''
    gen_hd = np.zeros((bin_digits.shape[0], 10))
    for i in range(10):
        for n in range(bin_digits.shape[0]):
            gen_hd[n][i] = (bin_digits[n] * np.log(eta[i]) + (1 - bin_digits[n]) * np.log(1 - eta[i])).sum()
    return gen_hd


def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    gene_likeihood = generative_likelihood(bin_digits, eta)
    cond = np.sum(np.exp(gene_likeihood - np.log(10)), axis=1).reshape((bin_digits.shape[0],))  # 700 x 1
    cond_standard = np.full((10, bin_digits.shape[0]), cond).T
    con_likelihood = (gene_likeihood - np.log(10)) - np.log(cond_standard)
    return con_likelihood

def avg_conditional_likelihood(data, labels, eta, stem):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''

    # Compute as described above and return
    cond_likelihood= conditional_likelihood(data, eta)
    sum_hd = 0
    for n in range(data.shape[0]):
        sum_hd += cond_likelihood[n][int(labels[n])]
    AVG = sum_hd / data.shape[0]
    print('Average conditional likelihood for ' + stem + 'data in correct class is: ', AVG)

    return


def classify_data(data, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = np.argmax(conditional_likelihood(data, eta), axis=1)
    # Compute and return the most likely class
    return cond_likelihood


def accuracy(labels, data, eta):
    return np.equal(labels, classify_data(data, eta)).mean()


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data',shuffle=False)
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)
    # Fit the model
    eta = compute_parameters(train_data.reshape((10,-1,64)))
    # Evaluation
    plot_images(eta)
    generate_new_data(eta)
    print('Train_data: ')
    avg_conditional_likelihood(train_data, train_labels, eta, data.TRAIN_STEM)
    print('Test_data: ')
    avg_conditional_likelihood(test_data, test_labels, eta, data.TEST_STEM)
    print('\nThe accuracy for train data is: ', accuracy(train_labels, train_data, eta))
    print('The accuracy for test data is: ', accuracy(test_labels, test_data, eta))


if __name__ == '__main__':
    main()
