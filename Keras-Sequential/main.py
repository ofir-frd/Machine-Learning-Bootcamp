
# Data handling tools
import numpy as np
import pandas as pd

# Pre-dl
from sklearn.model_selection import train_test_split

# Deep learning
from tensorflow import keras
from tensorflow.keras import layers


def main():

    df = pd.read_csv('diabetes.csv')
    x = df.drop('Outcome', axis=1)
    y = df['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    model = keras.Sequential()
    model.add(layers.Dense(8, activation='relu', input_shape=(8,)))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train.values, y_train.values, epochs=120, batch_size=32, verbose=2)

    _, accuracy_train = model.evaluate(x_train.values, y_train.values)
    print('Train accuracy: %.2f' % (accuracy_train * 100))

    y_predictions_test = (model.predict(x_test.values) > 0.5).astype(int)

    success_prediction_rate = np.equal(y_predictions_test.transpose(), y_test.values)
    count_true = np.count_nonzero(success_prediction_rate)
    print('Test accuracy: %.2f' % (100*count_true/len(y_test)))


if __name__ == '__main__':
    main()
