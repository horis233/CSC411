from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random


def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        # TODO: Plot feature i against y
        plt.scatter(X[:, i], y)
        plt.title('Target price and ' + features[i])
        plt.ylabel('Target price')
        plt.xlabel(features[i])
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    # TODO: implement linear regression
    n_training_samples = X.shape[0]
    n_dim = X.shape[1]
    x = np.reshape(np.c_[np.ones(n_training_samples), X], [n_training_samples, n_dim + 1])
    y = np.reshape(Y, [n_training_samples, 1])
    A = np.dot(x.T, x)
    B = np.dot(x.T, y)
    weights = np.linalg.solve(A, B)
    # Remember to use np.linalg.solve instead of inverting
    print(weights)
    return weights
    raise NotImplementedError()




def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))

    # Visualize the features
    visualize(X, y, features)

    # TODO: Split data into train and test

    my_list = [i for i in range(X.shape[0])]
    random.shuffle(my_list)
    test_num = int(X.shape[0] * 0.2)
    train_index = my_list
    test_index = []
    for i in range(test_num):
        test_index.append(my_list[i])
        del train_index[i]
    q1_x_train = X[train_index]
    q1_y_train = y[train_index]
    q1_x_test = X[test_index]
    q1_y_test = y[test_index]
    # Fit regression model
    w = fit_regression(q1_x_train, q1_y_train)
    # Compute fitted values, MSE, etc.
    x_test = np.reshape(np.c_[np.ones(q1_x_test.shape[0]), q1_x_test], [q1_x_test.shape[0], q1_x_test.shape[1] + 1])
    y_test = np.reshape(q1_y_test, [q1_x_test.shape[0], 1])
    q1_y_predicts = np.dot(x_test, w)
    plt.scatter(q1_y_test, q1_y_predicts)
    plt.xlabel("Prices: $Y_i$")
    plt.ylabel("Predicted prices: $\hat{Y}_i$")
    plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
    plt.show()
    # calculate RMSE MAE and MSE
    sum_mean = 0
    abs_mean = 0
    for i in range(len(q1_y_predicts)):
        sum_mean += (q1_y_predicts[i] - y_test[i]) ** 2
        abs_mean += abs(q1_y_predicts[i] - y_test[i])
    sum_error_rmse = np.sqrt(sum_mean / len(q1_y_predicts))
    sum_error_mse = sum_mean / len(q1_y_predicts)
    sum_error_mae = abs_mean / len(q1_y_predicts)
    print("MSE: ", sum_error_mse)
    print("RMSE: ", sum_error_rmse)
    print("MAE: ", sum_error_mae)

if __name__ == "__main__":
    main()
