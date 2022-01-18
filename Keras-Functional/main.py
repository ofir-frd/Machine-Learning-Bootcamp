
# Data handling tools
import numpy as np
import pandas as pd
from itertools import product

# Pre-machine learning tools
from sklearn.model_selection import train_test_split

# Deep learning
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def main():

    df_pixels = pd.read_csv('hmnist_8_8_L.csv')
    images_metadata = pd.read_csv('HAM10000_metadata.csv')

    current_lesion_id = images_metadata['lesion_id'].iloc[0]
    index_to_delete = []
    for idx, lesion_value in enumerate(images_metadata['lesion_id'][1:], start=1):

        if lesion_value == current_lesion_id:
            index_to_delete.append(idx)
            continue

        current_lesion_id = lesion_value

    df_pixels.drop(axis=0, index=index_to_delete, inplace=True)
    images_metadata.drop(axis=0, index=index_to_delete, inplace=True)

    x_1 = df_pixels.drop('label', axis=1)
    x_2 = images_metadata[['age', 'sex', 'localization']]
    x_2 = pd.get_dummies(x_2, prefix=['age', 'sex', 'localization'], columns=['age', 'sex', 'localization'])
    y = df_pixels['label'].map({0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0})

    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x_1, x_2, y, test_size=0.20,
                                                                             random_state=42)

    # define two sets of inputs
    input_a = keras.Input(shape=(64,))
    input_b = keras.Input(shape=(36,))

    # the first branch operates on the first input
    branch_a = layers.Dense(64, activation="relu")(input_a)
    branch_a = layers.Dense(32, activation="relu")(branch_a)
    branch_a = layers.Dense(16, activation="relu")(branch_a)
    branch_a = keras.Model(inputs=input_a, outputs=branch_a)

    # the second branch opreates on the second input
    branch_b = layers.Dense(36, activation="relu")(input_b)
    branch_b = layers.Dense(9, activation="relu")(branch_b)
    branch_b = keras.Model(inputs=input_b, outputs=branch_b)

    # combine the output of the two branches
    combined_branches = layers.concatenate([branch_a.output, branch_b.output])

    # combined outputs
    merged_network = layers.Dense(2, activation="relu")(combined_branches)
    merged_network = layers.Dense(1, activation="sigmoid")(merged_network)

    # Merged model
    model = keras.Model(inputs=[branch_a.input, branch_b.input], outputs=merged_network)

    print(model.summary())

    # Hyper-parameters
    learning_rate_list = [0.0001, 0.001, 0.01]
    loss_list = ['binary_crossentropy', 'hinge', 'poisson']
    metrics_list = ['accuracy', 'binary_accuracy', 'mse']
    epochs_list = [2, 5, 10]
    batch_size_list = [8, 32, 64]

    best_accuracy = 0
    for params in product(learning_rate_list, loss_list, metrics_list, epochs_list, batch_size_list):

        model.compile(optimizer=Adam(learning_rate=params[0]), loss=params[1], metrics=[params[2]])
        model.fit(x=[x1_train.values, x2_train.values], y=y_train.values, epochs=params[3], batch_size=params[4],
                  verbose=2)

        _, accuracy_train = model.evaluate(x=[x1_train.values, x2_train.values], y=y_train.values)
        print('Train accuracy: %.2f' % (accuracy_train * 100))

        if accuracy_train > best_accuracy:
            best_params = params

    print('Best Parameters:', best_params)

    model.compile(optimizer=Adam(learning_rate=best_params[0]), loss=best_params[1], metrics=[best_params[2]])
    model.fit(x=[x1_train.values, x2_train.values], y=y_train.values, epochs=best_params[3], batch_size=best_params[4],
              verbose=2)

    y_predictions_test = (model.predict([x1_test.values, x2_test.values]) > 0.5).astype(int)
    success_prediction_rate = np.equal(y_predictions_test.transpose(), y_test.values)
    count_true = np.count_nonzero(success_prediction_rate)
    print('Test accuracy: %.2f' % (100*count_true/len(y_test)))


if __name__ == '__main__':
    main()
