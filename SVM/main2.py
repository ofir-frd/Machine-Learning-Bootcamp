import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC


def main():

    # import some data to play with
    digits = datasets.load_digits()
    x_train = digits.data[0:900, :]
    x_test = digits.data[900: len(digits.data), :]
    y_train = digits.target[0:900]
    y_test = digits.target[900: len(digits.data)]

    figure, axis = plt.subplots(1, 1)
    i = 0
    for column in x_train:
        plt.imshow(np.reshape(column, (8, 8)))
        plt.show(block=False)
        i += 1
        if i == 10:
            break

    for c_value in range(1, 10, 1):
        svc = SVC(C=c_value/2)
        svc.fit(x_train, y_train)

        predictions = svc.predict(x_test)

        success_rate = np.where(predictions == y_test, 1, 0)

        print("success rate: {:.2f}".format(round(100*np.bincount(success_rate)[1]/len(success_rate), 2)))


if __name__ == '__main__':
    main()
