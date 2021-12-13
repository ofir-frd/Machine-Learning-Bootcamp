import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC


def main():
    # import some data to play with
    iris = datasets.load_iris()
    x_train = iris.data[:, [1,2]]  # we only take the first two features.
    y_train = iris.target

    x_test_synthetic = (np.array([np.random.random(4)+5, np.random.random(4)+3])).T

    for c_value in range(1, 30, 5):
        svc = SVC(C=c_value/10)
        svc.fit(x_train, y_train)

        predictions = svc.predict(x_test_synthetic)

        x_all = np.concatenate([x_train, x_test_synthetic])
        y_all = np.concatenate([y_train, predictions])

        color_condition = []

        for label in y_all:
            if label == 2:
                color_condition.append('green')
            elif label == 1:
                color_condition.append('blue')
            else:
                color_condition.append('red')

        # plot
        figure, axis = plt.subplots(1, 1)

        plt.scatter(x_all[:, 0], x_all[:, 1], color=color_condition)
        plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], color='black')
        axis.set_xlabel('sepal length (cm)')
        axis.set_ylabel('sepal length (cm)')

        plt.show(block=False)

    pass


if __name__ == '__main__':
    main()
