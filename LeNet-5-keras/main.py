
# Data handling tools
import numpy as np
import pandas as pd

# Deep learning
from tensorflow import keras
from tensorflow.keras import layers


def main():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    x_train_padded = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test_padded = x_test.reshape(x_test.shape[0], 28, 28, 1)

    '''
    LeNet-5 neural network architecture:
    #1 Convolutional  (tf.keras.layers.Conv2D)
    #2 Pooling  (tf.keras.layers.MaxPool2D)
    #3 Convolutional
    #4 Pooling
    #5 Flatten (make the results one dimensional so we can use fully connected layer on it)(tf.keras.layers.Flatten)
    #6 Fully connected with an output of 120 neurons (tf.add(tf.matmul(inX,w),b)
    #7 Fully connected with an output of 84 neurons
    #8 Fully connected with an output of 10 neurons
    '''

    model = keras.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=[3, 3], activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=48, kernel_size=[3, 3], padding='valid', activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=120, activation='relu'))
    model.add(layers.Dense(units=84, activation='relu'))
    model.add(layers.Dense(units=10, activation='softmax'))

    model.build()
    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train_padded, y_train.values, epochs=10, batch_size=32, verbose=2)

    _, accuracy_train = model.evaluate(x_train_padded, y_train.values)
    print('Train accuracy: %.2f' % (accuracy_train * 100))

    y_predictions_test = (model.predict(x_test_padded) > 0.5).astype(int)

    success_prediction_rate = np.equal(y_predictions_test, y_test.values)
    count_true = np.count_nonzero(success_prediction_rate)/10
    print('Test accuracy: %.2f' % (100*count_true/len(y_test)))


if __name__ == '__main__':
    main()
