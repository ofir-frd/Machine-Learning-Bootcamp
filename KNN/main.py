import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import voronoi_plot_2d, Voronoi, KDTree


# train-test split by a percentage
def split_df(user_df, split_ratio, random_value):
    x_train = user_df.sample(frac=split_ratio, random_state=random_value)
    x_test = user_df.drop(x_train.index)
    y_train = x_train['class']
    y_test = x_test['class']

    return x_train.drop('class', axis=1), x_test.drop('class', axis=1), y_train, y_test


# calculate manhattan distance between two point
def calculate_manhattan_distance(a, b):
    return np.sum([abs(var2 - var1) for var1, var2 in zip(a, b)])


# find knn for a point from a given data
def find_knn(point, data, k_value):
    index_list = data.index
    distances_list = []

    [distances_list.append(calculate_manhattan_distance(point, np.array(data.loc[row_index]))) for row_index in
     data.index]

    distances_df = pd.DataFrame(np.c_[index_list, distances_list], columns=['index', 'distances'])
    distances_df['index'] = distances_df['index'].apply(np.int64)
    distances_df = distances_df.set_index('index')
    distances_df = distances_df.sort_values('distances')

    return distances_df.index[np.arange(0, k_value, 1)]


# predict label of a point
def predict_label(point, data, labels, k_value):
    labels_list = []
    index_list = find_knn(point, data, k_value)

    [labels_list.append(labels[index]) for index in index_list]

    return np.bincount(labels_list).argmax()


# create line plot based of given color conditions
def plot_scatter(x, y, color_condition, labelx, labely, legend_title):
    figure, axis = plt.subplots(1, 1)

    plt.scatter(x, y, color=color_condition, label=legend_title)
    axis.set_xlabel(labelx)
    axis.set_ylabel(labely)
    axis.legend(ncol=2, loc='best', frameon=True)

    plt.show(block=False)


def main():
    iris_df = pd.read_csv('iris.csv')
    iris_df.columns = ['index', 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)',
                       'class']
    iris_df = iris_df.set_index('index')
    iris_df['class'] = [1 if (x == 'Iris-setosa') else 0 for x in iris_df['class']]

    x = iris_df[['sepal length (cm)', 'sepal width (cm)', 'class']].copy()

    # print(x.head())

    color_condition = ['blue' if (x == 1) else 'red' for x in x['class']]
    plot_scatter(x['sepal length (cm)'], x['sepal width (cm)'], color_condition,
                 'sepal length (cm)', 'sepal length (cm)', 'Iris-setosa')

    #############
    ### train ###
    #############
    x_train, x_test, y_train, y_test = split_df(iris_df, 0.1, 41)

    predictions = []

    [predictions.append(predict_label(np.array(x_train.loc[row_index]), x_train, y_train, 1)) for row_index in
     x_train.index]

    success_prediction_rate = np.equal(predictions, y_train)
    count_true = np.count_nonzero(success_prediction_rate)

    print('train: correct {} times out of {}, success rate of {}%'.format(
        count_true, len(y_train), round(100 * count_true / len(y_train), 1)))

    color_condition = ['blue' if (x == 1) else 'red' for x in y_train]
    plot_scatter(x_train['sepal length (cm)'], x_train['sepal width (cm)'], color_condition,
                 'sepal length (cm)', 'sepal length (cm)', 'Iris-setosa')

    ############
    ### test ###
    ############

    predictions = []

    [predictions.append(predict_label(np.array(x_test.loc[row_index]), x_train, y_train, 1)) for row_index in
     y_test.index]

    success_prediction_rate = np.equal(predictions, y_test)
    count_true = np.count_nonzero(success_prediction_rate)

    print('test: correct {} times out of {}, success rate of {}%'.format(
        count_true, len(y_test), round(100 * count_true / len(y_test), 2)))

    color_condition = ['blue' if (x == 1) else 'red' for x in predictions]
    plot_scatter(x_test['sepal length (cm)'], x_test['sepal width (cm)'], color_condition,
                 'sepal length (cm)', 'sepal length (cm)', 'Iris-setosa')

    points_train = np.transpose(np.array([x_train['sepal length (cm)'], x_train['sepal width (cm)']]))
    points_test = np.transpose(np.array([x_test['sepal length (cm)'], x_test['sepal width (cm)']]))
    vor = Voronoi(points_train)
    tree = KDTree(points_train)
    locs, ids = tree.query(points_test)

    fig, ax = plt.subplots(1, 1)
    voronoi_plot_2d(vor, ax)
    ax.scatter(x_train['sepal length (cm)'], x_train['sepal width (cm)'], s=20, color='r')

    plt.show()

    pass


if __name__ == '__main__':
    main()
