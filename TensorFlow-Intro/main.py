import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class MyModel:

    def __init__(self, weight: tf.Variable = 1, b_value: tf.Variable = 1, target_value: tf.Variable = 1,
                 predict_value: tf.Variable = 1, new_learning_rate: tf.Variable = 0.01):
        """
        Initiate MyModel class.
        :param weight: initial weight for linear regression.
        :param b_value: initial b value.
        :param target_value: y_target value for the regression.
        :param predict_value: y_predict, initial prediction.
        :param new_learning_rate: leaning rate of the algorithm.
        """
        self.W = weight
        self.b = b_value
        self.y_target = target_value
        self.y_predict = predict_value
        self.learning_rate = new_learning_rate

    def __call__(self, x, epochs_total):
        """
        Call the model.
        :param x: x values of the points.
        :param epochs_total: total optimization steps.
        :return: 3 lists: loss values, w and b per step.
        """
        return self.train(x, epochs_total)

    def calculate_loss(self):
        """
        Calculate the mean square value of two points.
        :return: mean square error loss.
        """
        return tf.reduce_mean((self.y_predict - self.y_target)**2)

    def train(self, x, epochs_total: int = 5):
        """
        Linear regression algorithm function.
        :param x: x values of the points.
        :param epochs_total:  total optimization steps.
        :return: 3 lists: loss values, w and b per step.
        """
        current_loss = []
        w_list = []
        b_list = []
        for i in range(epochs_total):

            with tf.GradientTape(persistent=True) as tape:
                self.y_predict = x @ self.W + self.b
                current_loss.append(self.calculate_loss())

            [dl_dw, dl_db] = tape.gradient(current_loss[-1], [self.W, self.b])

            self.W.assign_sub(self.learning_rate * dl_dw)
            self.b.assign_sub(self.learning_rate * dl_db)

            w_list.append(tf.identity(self.W))
            b_list.append(tf.identity(self.b))
            tf.print(current_loss[-1])

        return current_loss, w_list, b_list


def main():

    # Introduction:
    print(tf.__version__)

    node1 = tf.constant([1, 2, 3, 4, 5])
    node2 = tf.Variable([1, 1, 2, 3, 5])

    node3 = tf.math.multiply(node1, node2)

    print(node3)

    node4 = tf.reduce_sum(node3)

    print(node4)

    # Linear Regression:
    df = pd.read_csv('data_for_linear_regression_tf.csv')
    x = tf.constant(df['x'], dtype=tf.float32, shape=[699, 1])
    y = tf.constant(df['y'], dtype=tf.float32, shape=[699, 1])
    w = tf.Variable(tf.random.normal([1, 1], 0, 1, dtype=tf.float32, seed=1))
    b = tf.Variable(tf.zeros(1, dtype=tf.float32))
    learning_rate = tf.Variable(0.01)
    epochs = 100

    new_model = MyModel(w, b, y, tf.zeros(y.shape.as_list()[0], dtype=tf.float32), learning_rate)

    loss_values, w_values, b_values = new_model.__call__(x, epochs)
    w_extracted_values = np.squeeze([w_tensor.numpy() for w_tensor in w_values])
    b_extracted_values = np.squeeze([b_tensor.numpy() for b_tensor in b_values])

    # Plot:
    fig, axis = plt.subplots(figsize=(5, 5))

    plt.plot(np.arange(0, epochs, 1), loss_values, label='loss')
    plt.plot(np.arange(0, epochs, 1), w_extracted_values, label='w')
    plt.plot(np.arange(0, epochs, 1), b_extracted_values, label='b')
    axis.set_title('Loss, W and B Values')
    axis.set_xlabel("Epoch (num)")
    axis.set_ylabel("Value (num)")
    axis.legend()

    plt.show()


if __name__ == '__main__':
    main()
