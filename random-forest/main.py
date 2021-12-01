from Tree import *


# create new list of random features in root original length
def generate_features(features_list):

    new_random_features = features_list.to_numpy()
    np.random.shuffle(new_random_features)

    return new_random_features[0:int(np.sqrt(len(features_list)))]


# create a new sub dataframe of random samples
# size defined by ratio of original dataframe
def create_subsample(full_df, split_ratio):

    return full_df.sample(frac=split_ratio, random_state=np.random.randint(1, 50, 1))


# generate forest from given data and features list in requested size and depth
# returns arrays of trees (Root type)
def generate_forest(forest_df: pd.DataFrame, features_list, forest_size, trees_depth, bagging_ratio):

    new_forest = []

    forest_df_bootstrapped = pd.DataFrame(forest_df['label'])
    # bootstrapping the df
    for value in features_list:
        forest_df_bootstrapped[value] = forest_df[value]

    new_features_list = np.arange(-1, len(features_list), 1)
    forest_df_bootstrapped.columns = new_features_list
    forest_df_bootstrapped = forest_df_bootstrapped.rename(columns={-1: 'label'})

    for i in range(forest_size):

        # create a fresh forest_df_bootstrapped copy and get subsample of it (bagging)
        forest_df_subsample = forest_df_bootstrapped.copy()
        forest_df_subsample = create_subsample(forest_df_subsample, bagging_ratio)
        forest_df_subsample = forest_df_subsample.reset_index(drop=True)

        new_forest.append(build_tree(forest_df_subsample.drop('label', axis=1),
                                     pd.DataFrame(forest_df_subsample['label']), trees_depth))

    return new_forest


# generate predictions from all tree per row in test dataframe
def predict_forest_labels(random_forest, x_test):

    predictions = []

    for tree in random_forest:
        predictions.append(forest_predict_labels(tree, x_test))

    return predictions


# compute the label of each data row in a given test dataframe
# input: decision tree (root object) and test dataframe
# returns: an arrays of labels in length of test dataframe
def forest_predict_labels(decision_tree, x_test):

    predictions = []

    for index, row in x_test.iterrows():
        predictions.append(forest_find_value_and_get_label(decision_tree, list(zip(row, row.index))))

    return predictions


# recursively scan the branches of the tree. decide to take the left or right branch by existence of data or by
# appropriate values (current data in range of feature). find the optimum leaf by reaching the end of the line or by
# irrelevant branching.
def forest_find_value_and_get_label(node, row):

    if node.current_node.leaf == 1 or (node.current_node.left_node.empty and node.current_node.right_node.empty):
        return np.bincount(node.current_node.current_df['label']).argmax()

    elif not node.current_node.left_node.empty and node.current_node.right_node.empty:
        if row[node.current_node.feature][0] < node.current_node.left_node[node.current_node.feature].iloc[
                len(node.current_node.left_node) - 1]:
            return forest_find_value_and_get_label(node.left_node, row)
        else:
            return np.bincount(node.current_node.current_df['label']).argmax()

    elif node.current_node.left_node.empty and not node.current_node.right_node.empty:
        if row[node.current_node.feature][0] >= node.current_node.right_node[node.current_node.feature].iloc[0]:
            return forest_find_value_and_get_label(node.right_node, row)
        else:
            return np.bincount(node.current_node.current_df['label']).argmax()

    else:
        if row[node.current_node.feature][0] < node.current_node.left_node[node.current_node.feature].iloc[
                len(node.current_node.left_node) - 1]:
            return forest_find_value_and_get_label(node.left_node, row)
        if row[node.current_node.feature][0] >= node.current_node.right_node[node.current_node.feature].iloc[0]:
            return forest_find_value_and_get_label(node.right_node, row)

    return np.bincount(node.current_node.current_df['label']).argmax()


def main():

    # import and organizing the data
    bc_df = pd.read_csv('wdbc.data', names=np.arange(-2, 30, 1))
    bc_df = bc_df.rename(columns={-2: 'index', -1: 'label'})
    bc_df = bc_df.drop('index', axis=1)

    new_label = []
    for x in bc_df['label']:
        if x == 'M':
            new_label.append(1)
        else:
            new_label.append(0)

    bc_df['label'] = new_label

    x_train, x_test, y_train, y_test = split_df(bc_df, 'label', 0.8, 42)

    x_all = x_train.copy()
    x_all['label'] = y_train
    x_all = x_all.reset_index(drop=True)

    # amount of trees, their depth bagging ratio
    # forest_size = 5
    trees_depth = 2
    bagging_ratio = 0.5

    for i in range(0, 3, 1):

        # randomize features in root size of original quantity of features
        random_features = generate_features(x_train.columns)
        print('Features list:', random_features)

        for forest_size in range(1, 10, 1):

            # create a new forest_df_bootstrapped and subsample forest
            new_random_forest = generate_forest(x_all, random_features, forest_size, trees_depth, bagging_ratio)

            # update x_test to contain only relevant features
            x_test_reduced_features = pd.DataFrame()
            for value in random_features:
                x_test_reduced_features[value] = x_test[value]

            new_features_list = np.arange(0, len(random_features), 1)
            x_test_reduced_features.columns = new_features_list

            # received matrix of predictions in length of tree VS amount of tested samples.
            predictions_matrix = predict_forest_labels(new_random_forest, x_test_reduced_features)

            # find most common prediction per sample
            argmax_predictions_per_sample = [np.bincount(prediction_array).argmax()
                                             for prediction_array in zip(*predictions_matrix)]

            success_prediction_rate = np.equal(argmax_predictions_per_sample, y_test['label'])
            count_true = np.count_nonzero(success_prediction_rate)

            print('forest_size = {} max_depth = {}: correct {} times out of {}, success rate of {}%'.format(
                forest_size, trees_depth, count_true, len(y_test), round(100 * count_true / len(y_test), 2)))


if __name__ == '__main__':
    main()
