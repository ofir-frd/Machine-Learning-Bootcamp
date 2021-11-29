import numpy as np
import pandas as pd
from functools import lru_cache


# train-test split by a percentage.
# input: dataframe, label column name, split ration, and random state
# returns: x_train, x_test, y_train, y_test
def split_df(user_df, label_name, split_ratio=0.8, random_value=42):
    x_train = user_df.sample(frac=split_ratio, random_state=random_value)
    x_test = user_df.drop(x_train.index)

    return x_train.drop(label_name, axis=1), x_test.drop(label_name, axis=1), pd.DataFrame(
        x_train[label_name]), pd.DataFrame(x_test[label_name])


# splits the dataframe to two between by a given value in a specific feature
# input: dataframe, feature column name, and threshold split value
# returns: left_split - data with feature values lower then the threshold
# right_split - higher threshold values
def split_by_feature(user_df, feature_name, split_value):
    user_df = user_df.sort_values(feature_name, axis=0)

    left_split = user_df.iloc[0:split_value]
    right_split = user_df.iloc[split_value:-1]

    return left_split, right_split


# merge feature with label.
# order feature value from small to large (keep original row number) and generate gap values feature
# input: dataframe (x) and labels list (y)
# returns: DataFrame with new index ordered index old index and value order by size
# the ordered value are generated in between each data point of the given dataframe
def order_features(current_df: pd.DataFrame, feature: str, labels_list: pd.DataFrame):
    # create new dataframe with orders values and new index
    current_df['label'] = labels_list
    current_df = current_df.sort_values(feature, axis=0)
    ordered_df = current_df[[feature, 'label']]
    ordered_df = ordered_df.reset_index(drop=False)
    ordered_df = ordered_df.append(ordered_df.iloc[len(ordered_df) - 1], ignore_index=True)
    ordered_df['index'] = ordered_df['index'].astype(int)
    ordered_df['label'] = ordered_df['label'].astype(int)
    new_values = []

    for i in range(0, len(ordered_df)):

        if i == 0:
            new_values.append(ordered_df[feature].iloc[i] / 2)

        elif i == len(ordered_df) - 1:
            new_values.append((ordered_df[feature].iloc[i] + ordered_df[feature].iloc[i]) / 2)

        else:
            new_values.append((ordered_df[feature].iloc[i] + ordered_df[feature].iloc[i - 1]) / 2)

    ordered_df['averaged'] = new_values

    return ordered_df


# calculate gini index of the entire data frame and returns the position of minimum value
# input: dataframe (x) and labels list (y)
# returns: row number and column name and
def get_split(current_df: pd.DataFrame, labels_list: pd.DataFrame):
    # create an initial gini_matrix with 0.5 in each cell
    gini_matrix = np.ones((len(current_df) + 1, len(current_df.columns)))
    gini_matrix = gini_matrix - 0.5
    gini_matrix = pd.DataFrame(gini_matrix, columns=current_df.columns)

    # amount of rows in dataframe
    total_samples = len(current_df)

    # examine the data column be column
    for feature in current_df.columns:

        # order feature value from small to large (keep original row number)
        ordered_features = order_features(current_df, feature, labels_list)

        # examine rows in column
        for current_position in range(0, len(ordered_features)):

            # count the amount of 1 labels from start to current label
            counter_before = 0
            for i in range(0, current_position):
                if ordered_features['label'].iloc[i] == 1:
                    counter_before += 1

            # count the amount of 1 labels from current label to end
            counter_after = 0
            for i in range(current_position + 1, total_samples):
                if ordered_features['label'].iloc[i] == 1:
                    counter_after += 1

            # calculate ratio of 1, 0 and the gini of the data located before the current position
            if current_position == 0:
                proportion_before_1 = counter_before
            else:
                proportion_before_1 = counter_before / current_position
            proportion_before_0 = 1 - proportion_before_1
            gini_before = 1 - (proportion_before_1 ** 2 + proportion_before_0 ** 2)

            # calculate ratio of 1, 0 and the gini of the data located after the current position
            if total_samples - (current_position + 1) == 0:
                proportion_after_1 = counter_after
            else:
                proportion_after_1 = counter_after / (total_samples - (current_position + 1))
            proportion_after_0 = 1 - proportion_after_1
            gini_after = 1 - (proportion_after_1 ** 2 + proportion_after_0 ** 2)

            # calculate and update the gini matrix cell with the final gini value
            gini_matrix.loc[current_position, feature] = abs(gini_before * (
                                                         current_position + 1) / total_samples) + abs(
                                                         gini_after * (1 - ((current_position + 1) / total_samples)))

    row, column = gini_matrix.stack().idxmin()
    ordered_feature = order_features(current_df, column, labels_list)

    return int(ordered_feature.iloc[row]['index']), column  # returns: row number, column name, and gini value


# Decision tree node
class Node:
    left_node: pd.DataFrame
    right_node: pd.DataFrame
    current_df: pd.DataFrame
    feature: int
    row: int
    depth: int
    leaf: int
    labels: np.ndarray

    def __init__(self, current_df=pd.DataFrame(), depth=0):
        self.current_df = current_df
        self.feature = 0
        self.row = 0
        self.depth = depth
        self.leaf = 0
        self.left_node = pd.DataFrame()
        self.right_node = pd.DataFrame()
        self.labels = np.zeros(2, dtype=int)
        self.major_label = 0

    def split_data(self):

        self.labels = np.bincount(self.current_df['label'])
        self.major_label = self.labels.argmax()

        if self.leaf == 0:

            if self.current_df.empty:
                self.leaf = 1

            else:

                self.current_df = self.current_df.reset_index(drop=True)
                self.row, self.feature = get_split(self.current_df.drop('label', axis=1),
                                                   pd.DataFrame(self.current_df['label']))
                self.left_node, self.right_node = split_by_feature(self.current_df, self.feature, self.row)

        if self.left_node.empty and self.right_node.empty:
            self.leaf = 1


# holds a junction and two branches
class Root:
    current_node: Node
    left_node: Node
    right_node: Node

    def __init__(self):
        self.current_node = Node()
        self.left_node = Node()
        self.right_node = Node()


# Build a decision tree
def build_tree(x_train, y_train, max_depth):
    x_all = x_train
    x_all['label'] = y_train

    root = Root()
    root.current_node = Node(x_all, max_depth)

    root.left_node, root.right_node = split_node(root.current_node, max_depth)

    return root


# recursive function:
# look for the best split point of the available data
# check if we need to stop (Node.leaf is True), if leaf return it
# else, split while checking max depth
# if reached max depth return leaf
@lru_cache(maxsize=None)
def split_node(current_node: Node, depth: int):
    if current_node.leaf == 1:
        return current_node, current_node

    elif depth != 0:

        if current_node.current_df.empty:
            return current_node, current_node

        if len(np.bincount(current_node.current_df['label'])) == 1:
            return pd.DataFrame(), pd.DataFrame()

        current_node.split_data()

        left_node = Root()
        left_node.current_node.current_df = current_node.left_node
        left_node.left_node, left_node.right_node = split_node(left_node.current_node, depth - 1)

        right_node = Root()
        right_node.current_node.current_df = current_node.right_node
        right_node.left_node, right_node.right_node = split_node(right_node.current_node, depth - 1)

        return left_node, right_node

    else:
        current_node.leaf = 1
        return current_node, current_node


# compute the label of each data row in a given test dataframe
# input: decision tree (root object) and test dataframe
# returns: an arrays of labels in length of test dataframe
def predict_labels(decision_tree, x_test):
    predictions = []

    for index, row in x_test.iterrows():
        predictions.append(find_value_and_get_label(decision_tree, list(zip(row, row.index))))

    return predictions


# recursively scan the branches of the tree. decide to take the left or right branch by existence of data or by
# appropriate values (current data in range of feature). find the optimum leaf by reaching the end of the line or by
# irrelevant branching.
@lru_cache(maxsize=None)
def find_value_and_get_label(node, row):
    if node.current_node.leaf == 1:
        return np.bincount(node.current_node.current_df['label']).argmax()

    elif not node.current_node.left_node.empty and node.current_node.right_node.empty:
        if row[node.current_node.feature - 2][0] < node.current_node.left_node[node.current_node.feature].iloc[
                len(node.current_node.left_node) - 1]:
            return find_value_and_get_label(node.left_node, row)
        else:
            return np.bincount(node.current_node.current_df['label']).argmax()

    elif node.current_node.left_node.empty and not node.current_node.right_node.empty:
        if row[node.current_node.feature - 2][0] >= node.current_node.right_node[node.current_node.feature].iloc[0]:
            return find_value_and_get_label(node.right_node, row)
        else:
            return np.bincount(node.current_node.current_df['label']).argmax()

    else:
        if row[node.current_node.feature - 2][0] < node.current_node.left_node[node.current_node.feature].iloc[
                len(node.current_node.left_node) - 1]:
            return find_value_and_get_label(node.left_node, row)
        if row[node.current_node.feature - 2][0] >= node.current_node.right_node[node.current_node.feature].iloc[0]:
            return find_value_and_get_label(node.right_node, row)

    return np.bincount(node.current_node.current_df['label']).argmax()

'''
def main():
    bc_df = pd.read_csv('wdbc.data', names=np.arange(0, 32, 1))
    bc_df = bc_df.rename(columns={0: 'index', 1: 'label'})
    bc_df = bc_df.drop('index', axis=1)

    new_label = []
    for x in bc_df['label']:
        if x == 'M':
            new_label.append(1)
        else:
            new_label.append(0)

    bc_df['label'] = new_label

    
   # dataset = [[2.771244718, 1.784783929, 0],
               [1.728571309, 1.169761413, 0],
               [3.678319846, 2.81281357, 0],
               [3.961043357, 2.61995032, 0],
               [2.999208922, 2.209014212, 0],
               [7.497545867, 3.162953546, 1],
               [9.00220326, 3.339047188, 1],
               [7.444542326, 0.476683375, 1],
               [10.12493903, 3.234550982, 1],
               [6.642287351, 3.319983761, 1]]

    #small_df = pd.DataFrame(dataset, columns=['x','y','label'])
   #x_train, x_test, y_train, y_test = split_df(small_df, 'label', 0.8, 42)
    
    x_train, x_test, y_train, y_test = split_df(bc_df, 'label', 0.3, 42)

    for max_depth in range(2, 8):
        new_tree = build_tree(x_train, y_train, max_depth)

        predictions = predict_labels(new_tree, x_test)

        success_prediction_rate = np.equal(predictions, y_test['label'])
        count_true = np.count_nonzero(success_prediction_rate)

        print('test (max_depth = {}): correct {} times out of {}, success rate of {}%'.format(
            max_depth, count_true, len(y_test), round(100 * count_true / len(y_test), 2)))


if __name__ == '__main__':
    main()
'''