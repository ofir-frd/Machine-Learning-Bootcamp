"""

Definition:

q - points_probability_matrix
sigma - clusters_variance
mu - clusters_means

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def calculate_normal_distribution(point: np.ndarray = 0, clusters_means: np.ndarray = 0,
                                  clusters_variance: np.ndarray = 0):
    """ calculate the normal distribution of a point in a cluster
    # Arguments
        point: position of a point
        clusters_means: position of a cluster's center
        clusters_variance: variance of the cluster
    # Output
        normal distribution of a point
    """

    part_a = 1./((2 * np.pi)**2) * (np.sqrt(np.abs(np.linalg.det(clusters_variance))))
    part_b = - 0.5 * ((point - clusters_means).reshape(1, 4) @ np.linalg.inv(clusters_variance) @ (point - clusters_means).reshape(4, 1))
    if part_b > 0:
        part_b = -part_b
    part_b = np.exp(part_b)

    return part_a * part_b


def calculate_bayes_probability(df: np.ndarray = 0, clusters_means: np.ndarray = 0, clusters_variances: np.ndarray = 0,
                                clusters_weights: np.ndarray = 0, k: int = 0):
    """ calculate the normalized Bayes probability of each point per all clusters
    # Arguments
        df: arrays of points
        clusters_means: position of a cluster's center
        clusters_variance: variance of the cluster
        clusters_weights: weight of each cluster
        k: amount of clusters
    # Output
        2-D matrix of normalized Bayes probabilities. Each row represents a point, and each column represents a cluster
    """
    points_probability_matrix = np.zeros((150, k))

    for idx, point in enumerate(df):
        for cluster_idx, cluster_mean in enumerate(clusters_means):
            # calculates initial not normalized probability:
            points_probability_matrix[idx, cluster_idx] = clusters_weights[cluster_idx] * calculate_normal_distribution(point, cluster_mean, clusters_variances[cluster_idx])

        # normalize probabilities
        points_probability_matrix[idx, :] = clusters_weights * points_probability_matrix[idx, :] / (np.sum(points_probability_matrix[idx, :])+10**-10)

    return points_probability_matrix


def calculate_variances(df: np.ndarray = 0, clusters_means: np.ndarray = 0, points_probability_matrix: np.ndarray = 0,
                        points_assignment: np.ndarray = 0, k: int = 0):
    """ calculate variances of each cluster point
    # Arguments
        df: arrays of points
        clusters_means: arrpy of cluster centers
        points_probability_matrix: normalized Bayes probabilities
        assigned_points: assignments of points to clusters
        k: amount of clusters
    # Output
        distance: variances to each cluster
    """

    variance_of_clusters = np.zeros((k, 4, 4))

    for cluster_idx, cluster_mean in enumerate(clusters_means):

        part_a = 1/np.bincount(points_assignment == cluster_idx)[1]

        point = df[np.where(points_assignment == cluster_idx)]
        probabilities_per_cluster = points_probability_matrix[np.where(points_assignment == cluster_idx), cluster_idx]

        part_b = sum(probabilities_per_cluster.T*(point - cluster_mean)).reshape(1, 4) * sum(point - cluster_mean).reshape(4, 1)

        variance_of_clusters[cluster_idx] = part_a * part_b

    return variance_of_clusters


def calculate_new_means(df: np.ndarray = 0, points_probability_matrix: np.ndarray = 0, points_assignment: np.ndarray = 0, k: int = 0):
    """ calculate mean of each cluster point
    # Arguments
        df: arrays of points
        points_probability_matrix: normalized Bayes probabilities
        cluster_centers: arrpy of cluster centers
        k: amount of clusters
    # Output
        distance: mean value to each cluster
    """

    sum_of_clusters = np.zeros((k, 4))

    for idx, point in enumerate(df):
        sum_of_clusters[points_assignment[idx]] += point * points_probability_matrix[idx, points_assignment[idx]]

    return sum_of_clusters/np.bincount(points_assignment).reshape(3, 1)


def main():
    iris = load_iris()
    df = iris.data

    # Initialization:
    k = 3  # Amount of clusters
    random_points = np.random.randint(0, len(df), k)  # Random indexes generator

    clusters_means = df[[random_points]]  # Assign initial position to each cluster

    clusters_variances = [np.eye(4)] * k  # Initiate clusters variance

    clusters_weights = np.ones(k) * 1 / k  # Equally distributed wights

    points_assignment = np.zeros(len(df))  # Final assignment decision of points to clusters

    log_likelihoods = []

    epoch = 3  # number to iterations
    i = 0  # epoch counter
    while i < epoch:

        # Expectation:
        # update points probabilities for all clusters
        points_probability_matrix = calculate_bayes_probability(df, clusters_means, clusters_variances, clusters_weights, k)

        # calculate log_likelihoods from points_probability_matrix
        log_likelihoods.append(np.sum(np.log(np.sum(points_probability_matrix, axis=1)+0.000001)))

        points_assignment = np.argmax(points_probability_matrix, axis=1)

        # Maximization:
        # The new weights
        clusters_weights = (1/len(df)) * np.bincount(points_assignment)

        # calculate new clusters means
        clusters_means = calculate_new_means(df, points_probability_matrix, points_assignment, k)

        # calculate new clusters variance
        clusters_variances = calculate_variances(df, clusters_means, points_probability_matrix, points_assignment, k)

        # Plot:
        figure, axis = plt.subplots(1, 1)

        plt.scatter(df[:, 0], df[:, 1], c=points_assignment, cmap='brg')
        axis.set_xlabel('sepal length (cm)')
        axis.set_ylabel('sepal length (cm)')
        plt.text(3.8, 4.6, ['log likelihoods values: ', log_likelihoods])
        plt.show()

        if len(log_likelihoods) > 2:
            if (log_likelihoods[-1] - log_likelihoods[-2]) < 0.0001:
                print('Model stopped: log_likelihood low value')
                break

        i += 1


if __name__ == '__main__':
    main()
