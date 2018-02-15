'''
Question 1 Skeleton Code


'''

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import matplotlib.pyplot as plt


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test


def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data)  # bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names()  # converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names


def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data)  # bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names()  # converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names


def bnb_baseline(train, train_labels, test, test_labels):
    # training the baseline model
    binary_train = (train > 0).astype(int)
    binary_test = (test > 0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    # evaluate the baseline model
    train_pred = model.predict(train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def sgd_classifier(train, train_labels, test, test_labels):
    # training the baseline model

    model = SGDClassifier()
    #model.fit(train, train_labels)
    param_grid = dict(max_iter=np.arange(34, 38, 0.1))
    model_best = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_iter=10, n_jobs=-1)
    model_best.fit(train, train_labels)

    # evaluate the baseline model
    train_pred = model_best.predict(train)
    print('SGD Classifier train accuracy = {} with hyper-parameter {}'.format((train_pred == train_labels).mean(), model_best.best_params_))
    test_pred = model_best.predict(test)
    print('SGD Classifier test accuracy = {}with hyper-parameter {}'.format((test_pred == test_labels).mean(), model_best.best_params_))

    return model


def logistic_regression(train, train_labels, test, test_labels):
    # training the baseline model

    model = LogisticRegression()
    #model.fit(train, train_labels)
    param_grid = dict(C=np.arange(0.2, 0.4, 0.01))
    model_best = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_iter=10, random_state=5, n_jobs=-1)
    model_best.fit(train, train_labels)
    # evaluate the baseline model
    train_pred = model_best.predict(train)
    print('Logistic Regression train accuracy = {} with hyper-parameter {}'.format((train_pred == train_labels).mean(), model_best.best_params_))
    test_pred = model_best.predict(test)
    print('Logistic Regression test accuracy = {} with hyper-parameter {}'.format((test_pred == test_labels).mean(), model_best.best_params_))

    return model


def Multi_NB(train, train_labels, test, test_labels):

    model = MultinomialNB()
    model.fit(train, train_labels)

    # TODO
    # hyper-param tuning
    # evaluate the logistic regression model
    param_grid = dict(alpha=np.geomspace(0.0001, 0.1, 50))
    model_best = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    model_best.fit(train, train_labels)
    train_pred = model_best.predict(train)
    print('Multinomial Naive Bayes train accuracy  = {} with hyper-parameter {}'.format((train_pred == train_labels).mean(), model_best.best_params_))
    test_pred = model_best.predict(test)
    print('Multinomial Naive Bayes test accuracy = {} with hyper-parameter {}'.format((test_pred == test_labels).mean(), model_best.best_params_))
    matrix = compute_confusion_matrix(test_pred, test_data.target)
    plt.imshow(matrix, cmap='gray')
    plt.colorbar()
    plt.show()
    return model


def find_max(c_matrix):
    max_one = np.argmax(c_matrix)
    max_row = (max_one + 1) // 20
    max_col = max_one % 20
    c_matrix[max_row][max_col] = -1
    print("{} is misdiagnosed as {} ".format(max_col, max_row))
    return c_matrix


def compute_confusion_matrix(test_predictions, test_targets):
    confusion_matrix = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            for n in range(len(test_predictions)):
                if test_predictions[n] == i and test_targets[n] == j:
                    confusion_matrix[i][j] += 1
    matrix_d = confusion_matrix
    np.fill_diagonal(matrix_d, 0)

    matrix_t = find_max(matrix_d)
    # now discard this let's find the next max
    find_max(matrix_t)

    return matrix_d


if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    sgd_model = sgd_classifier(train_bow, train_data.target, test_bow, test_data.target)
    lr_model = logistic_regression(train_bow, train_data.target, test_bow, test_data.target)
    mnb_model = Multi_NB(train_bow, train_data.target, test_bow, test_data.target)

