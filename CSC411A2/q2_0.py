'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def plot_means(train_data):

    one_digit = 700
    mean = np.zeros((10, 64))
    for i in range(10):
        mean[i] = np.mean(train_data[i*one_digit:(i+1)*one_digit], axis=0)
        # Compute mean of class

    for i in range(0, 10):
        plt.subplot(2, 5, i+1)
        plt.imshow(mean[i].reshape((8, 8)), cmap='gray')

    # Plot all means on same axis
    plt.tight_layout()
    plt.show()

def main():

    train_data, _, _, _ = data.load_all_data_from_zip('a2digits.zip', 'data', shuffle=False)
    plot_means(train_data)

if __name__ == '__main__':
    main()

