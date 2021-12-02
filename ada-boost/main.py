import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import plotly.express as px


# train-test split by a percentage.
# input: dataframe, label column name, split ration, and random state
# returns: x_train, x_test, y_train, y_test
def split_df(user_df: pd.DataFrame, label_name: str, split_ratio=0.8, random_value=42):
    x_train = user_df.sample(frac=split_ratio, random_state=random_value)
    x_test = user_df.drop(x_train.index)

    return x_train.drop(label_name, axis=1), x_test.drop(label_name, axis=1), pd.DataFrame(
        x_train[label_name]), pd.DataFrame(x_test[label_name])


# import data and preprocess it
def preprocessing(file_name: str):

    # data import
    heart_df = pd.read_csv(file_name)

    # converting target to 1 and -1
    new_label = []
    for x in heart_df['target']:
        if x == 1:
            new_label.append(1)
        else:
            new_label.append(-1)

    heart_df['target'] = new_label
    # heart_df = heart_df.rename(columns={'target': 'label'})

    # hot encoding of relevant features
    dummy_features_list = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    non_dummy_features_list = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

    new_heart_df = pd.DataFrame(heart_df[non_dummy_features_list])
    for feature in dummy_features_list:
        new_heart_df = new_heart_df.join(pd.get_dummies(heart_df[feature], prefix=feature))

    return heart_df


# Create as arrays of stump tree in a given size
def create_stump_forest(forest_size: int, random_state_local: int):

    stump_forest = []
    for i in range(0, forest_size, 1):
        stump_forest.append(DecisionTreeClassifier(criterion='gini', max_depth=1, random_state=random_state_local))

    return stump_forest


# update weight of each row and randomly generate a new weighted data frame
# input: x/y data, predictions list, current stump weight
# return: new weighted x and y data frames
def create_new_weighted_data(x: pd.DataFrame, y: pd.DataFrame, predictions: np.ndarray, stump_weight: list):

    # initiate weights
    sample_weight = 1/len(x)
    new_weights = []

    # calculate new weights based on correct and incorrect decisions
    for i in range(0, len(predictions), 1):
        if predictions[i] == 1:
            new_weights.append(sample_weight*np.exp(-np.sum(stump_weight)))
        else:
            new_weights.append(sample_weight*np.exp(np.sum(stump_weight)))

    # normalize weights
    sum_of_new_weights = sum(new_weights)
    new_normalized_weights = new_weights/sum_of_new_weights

    # create normalized distributions weights for random rows pulling
    distribution_weights = []
    accumulator = 0
    for new_normalized_weight in new_normalized_weights:
        accumulator += new_normalized_weight
        distribution_weights.append(accumulator)

    # based to rows weights values, randomly pick new data
    new_x = pd.DataFrame(columns=x.columns)
    new_y = pd.DataFrame(columns=y.columns)
    array_of_distributions = np.asarray(distribution_weights)  # transform list to array for np usage

    for i in range(0, len(array_of_distributions), 1):
        random_number = np.random.uniform(0, 1, 1)
        index_of_row = (np.abs(array_of_distributions - random_number)).argmin()
        if array_of_distributions[index_of_row] < random_number and index_of_row < len(x)-1:
            index_of_row += 1

        x_new_row = pd.DataFrame(x.iloc[index_of_row]).T
        y_new_row = pd.DataFrame(y.iloc[index_of_row]).T
        new_x = pd.concat([new_x, x_new_row])
        new_y = pd.concat([new_y, y_new_row])

    # reset rows index to evert duplications
    new_x = new_x.reset_index(drop=True)
    new_y = new_y.reset_index(drop=True)
    new_y['target'] = new_y['target'].astype(int)  # correct y target from object to int

    return new_x, new_y


# train the stump forest
# arrays of untrained stumps forest, initial weighted x and y train
# return: predictions matrix of all samples per stump, stump weight array for test data
def train_stump_forest(stump_forest: list, weighted_x_train: pd.DataFrame, weighted_y_train: pd.DataFrame):

    # train stump forest
    stump_weight = []  # weight of all stumps for final decision.
    predictions_matrix = []

    for stump in stump_forest:
        # train the current stump
        stump.fit(weighted_x_train, weighted_y_train)

        # predict results based on its training
        predictions = stump.predict(weighted_x_train)
        predictions_matrix.append(predictions)

        # stump weight score to be used to final forest decision
        error_occurrences = np.count_nonzero(predictions == -1)
        total_error = error_occurrences * (1 / len(weighted_x_train))
        new_stump_weight = 0.5 * np.log(((1 - total_error)+0.00001) / (total_error+0.00001))
        stump_weight.append(new_stump_weight)

        # update samples weights and create new dataset
        weighted_x_train, weighted_y_train = create_new_weighted_data(weighted_x_train, weighted_y_train, predictions,
                                                                      stump_weight)

    return predictions_matrix, stump_weight


# make final decision on each sample based on all stumps decisions
def make_final_prediction(predictions_matrix: list, stump_weight: list):

    final_predictions = []

    for row_of_stumps_prediction in zip(*predictions_matrix):

        grade_positive = 0
        grade_negative = 0

        for i in range(0, len(row_of_stumps_prediction), 1):
            if row_of_stumps_prediction[i] == 1:
                grade_positive += stump_weight[i]
            else:
                grade_negative += stump_weight[i]

        if grade_positive > grade_negative:
            final_predictions.append(1)
        else:
            final_predictions.append(-1)

    return final_predictions


def main():

    # data import and preprocessing
    heart_df = preprocessing('heart.csv')

    multi_hearts = pd.concat([heart_df, heart_df, heart_df, heart_df], ignore_index=True)
    # splitting of the data
    x_train, x_test, y_train, y_test = split_df(multi_hearts, 'target', 0.8, 42)

    # setting up stump forest object:
    success_rates_train = []
    success_rates_test = []
    amount_of_trees_list = np.arange(1, 50, 4)
    for amount_of_trees in amount_of_trees_list:

        forest_size = amount_of_trees
        random_state_local = 42
        stump_forest = create_stump_forest(forest_size, random_state_local)

        # initiate weighted database
        weighted_x_train, weighted_y_train = x_train, y_train

        #%% Train

        # weight of all stumps for final decision
        predictions_matrix, stump_weight = train_stump_forest(stump_forest, weighted_x_train, weighted_y_train)

        final_predictions = make_final_prediction(predictions_matrix, stump_weight)
        success_prediction_rate = np.equal(final_predictions, y_train['target'])
        count_true = np.count_nonzero(success_prediction_rate)
        success_rate = round(100 * count_true / len(y_train), 2)
        success_rates_train.append(success_rate)
        print('train: number of stumps {}, correct {} times out of {}, success rate of {}%'.format(
            forest_size, count_true, len(y_train), success_rate))

        #%% Test

        predictions_matrix = []
        for stump in stump_forest:
            predictions = stump.predict(x_test)
            predictions_matrix.append(predictions)

        final_predictions = make_final_prediction(predictions_matrix, stump_weight)
        success_prediction_rate = np.equal(final_predictions, y_test['target'])
        count_true = np.count_nonzero(success_prediction_rate)
        success_rate = round(100 * count_true / len(y_test), 2)
        success_rates_test.append(success_rate)
        print('test: number of stumps {}, correct {} times out of {}, success rate of {}%'.format(
            forest_size, count_true, len(y_test), success_rate))


    #%% plot success_rate vs forest size
    results = pd.DataFrame(data=amount_of_trees_list, columns=['amount_of_trees'])
    results['Train'] = success_rates_train
    results['Test'] = success_rates_test
   # results = results.set_index('amount_of_trees')
    fig = px.scatter(results, x='amount_of_trees', y=['Train','Test'], size='amount_of_trees')
    fig.update_layout(xaxis_title="Amount of Trees (num.)", yaxis_title="Success Rate (%)")
    fig.show()
    print(stump_weight)


if __name__ == '__main__':
    main()
