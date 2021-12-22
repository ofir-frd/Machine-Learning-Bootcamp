import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class Cluster:

    def __init__(self, starting_points=None, new_df=None):
        if starting_points is None:
            starting_points = []
        if new_df is None:
            new_df = []
        self.points_idx: np.ndarray = starting_points
        self.df: np.ndarray = new_df

    def set_points(self, new_points_idx: np.ndarray):
        """ update the cluster points list
        # Arguments
            new_points_idx: list
        # Output
            none
        """
        self.points_idx = new_points_idx

    def get_points(self):
        return self.points_idx

    def add_point(self, point: int = 0):
        self.points_idx.append(point)

    def calculate_complete_linkage_criterion(self, point):
        """ Calculate distance between two points
                # Arguments
                    point_a, point_b: np.ndarray
                # Output
                    distance: float
        """

        distance = 0

        for index in self.points_idx:
            new_distance = self.calculate_distance(self.df[index], point)
            if new_distance > distance:
                distance = new_distance

        return distance

    @staticmethod
    def calculate_distance(point_a: np.ndarray, point_b: np.ndarray):
        """ Calculate distance between two points
        # Arguments
            point_a, point_b: np.ndarray
        # Output
            distance: float
        """
        distance = 0

        for idx, value_a in enumerate(point_a):
            distance += (value_a-point_b[idx])**2

        return np.sqrt(distance)


def find_distance_between_clusters(df, cluster_a: Cluster, cluster_b: Cluster):
    """ Calculate distance between two clusters under full linkage condition
    # Arguments
        Two clusters objects
    # Output
        distance: float
    """
    distance = 0

    for index in cluster_b.get_points:
        new_distance = cluster_a.calculate_complete_linkage_criterion(df[index])
        if new_distance > distance:
            distance = new_distance

    return distance


def calculate_distance_map(df: np.ndarray):
    """ Calculate distance between all points
    # Arguments
        np array of points, size (n,m)
    # Output
        distance: np arrays of distances, size (n,m)
    """
    distance_map = []
    for current_idx, current_item in enumerate(df):
        for other_idx, other_item in enumerate(df):
            distance_map.append(Cluster.calculate_distance(current_item, other_item))

    return np.reshape(distance_map, (len(df), len(df)))


def find_closest_couple(distance_map: np.ndarray = 0):

    zeros_removed = distance_map[distance_map != 0]  # removal of all zeros
    min_distance = np.nanmin(zeros_removed)  # find minimum value
    min_index = np.where(distance_map == min_distance)  # get index of minimum value

    return min_index


def do_clustering(df: np.ndarray, idx_list: np.ndarray, distance_map: np.ndarray = 0, cluster_list=None):

    # todo: compare all point and clusters
    #  decide if to merge clusters/add point to cluster/merge points to new cluster.
    #  create/update cluster and delete from new_df

    closest_couple_index = find_closest_couple(distance_map)  # find location of the closest couple

    if cluster_list is None:
        cluster_list = [Cluster(closest_couple_index[0], df)]
        # delete cells
        idx_list = np.delete(idx_list, [closest_couple_index[0][0], closest_couple_index[0][1]], None)
        # delete rows
        distance_map = np.delete(distance_map, [closest_couple_index[0][0], closest_couple_index[0][1]], 0)
        # delete columns
        distance_map = np.delete(distance_map, 0, [closest_couple_index[0][0], closest_couple_index[0][1]])
        # add new cluster distance
        new_distances = []

        for idx in idx_list:
            new_distances.append(cluster_list[0].calculate_complete_linkage_criterion(df[idx]))

        new_distances.append(0)
        distance_map = np.r_[distance_map, new_distances]
        distance_map = np.c_[distance_map, new_distances]

    else:

        # creating new cluster if closest couple is between points
        if closest_couple_index[0][0] < idx_list[len(idx_list)] and closest_couple_index[0][1] < idx_list[len(
                idx_list)]:
            cluster_list.append(Cluster(closest_couple_index[0], df))
            # delete cells
            idx_list = np.delete(idx_list, [closest_couple_index[0][0], closest_couple_index[0][1]], None)
            # delete rows
            distance_map = np.delete(distance_map, [closest_couple_index[0][0], closest_couple_index[0][1]], 0)
            # delete columns
            distance_map = np.delete(distance_map, 0, [closest_couple_index[0][0], closest_couple_index[0][1]])

            # add new cluster distance
            new_distances = []

            for idx in idx_list:
                new_distances.append(cluster_list[0].calculate_complete_linkage_criterion(df[idx]))

            for cluster in cluster_list:
                new_distances.append(find_distance_between_clusters(cluster, cluster_list[len(cluster_list)]))

            distance_map = np.r_[distance_map, new_distances]
            distance_map = np.c_[distance_map, new_distances]

        # merge cluster and item
        else:

            # merge point with cluster if the 1st items is a point:
            if closest_couple_index[0][0] < idx_list[len(idx_list)]:
                # add point to cluster
                cluster_list[len(distance_map)-closest_couple_index[0][1]].add_point(
                    idx_list[closest_couple_index[0][0]])
                # delete cell
                idx_list = np.delete(idx_list, closest_couple_index[0][0], None)
                # delete rows
                distance_map = np.delete(distance_map, closest_couple_index[0][0], 0)
                # delete columns
                distance_map = np.delete(distance_map, 0, closest_couple_index[0][0])

                # update cluster distance
                new_distances = []

                for idx in idx_list:
                    new_distances.append(cluster_list[0].calculate_complete_linkage_criterion(df[idx]))

                for cluster in cluster_list:
                    new_distances.append(find_distance_between_clusters(cluster, cluster_list[len(cluster_list)]))

                distance_map[0][closest_couple_index[0][1]] = new_distances
                distance_map[closest_couple_index[0][1]][0] = new_distances

            # merge point with cluster if the 2nd items is a point:
            if closest_couple_index[0][1] < idx_list[len(idx_list)]:
                # add point to cluster
                cluster_list[len(distance_map) - closest_couple_index[0][0]].add_point(
                    idx_list[closest_couple_index[0][1]])
                # delete cell
                idx_list = np.delete(idx_list, closest_couple_index[0][1], None)
                # delete rows
                distance_map = np.delete(distance_map, closest_couple_index[0][1], 0)
                # delete columns
                distance_map = np.delete(distance_map, 0, closest_couple_index[0][1])

                # update cluster distance
                new_distances = []

                for idx in idx_list:
                    new_distances.append(cluster_list[0].calculate_complete_linkage_criterion(df[idx]))

                for cluster in cluster_list:
                    new_distances.append(find_distance_between_clusters(cluster, cluster_list(len(cluster_list))))

                distance_map[0][closest_couple_index[1][0]-1] = new_distances
                distance_map[closest_couple_index[1][0]][0-1] = new_distances

            # merge cluster with cluster
            else:
                # add indexes of one cluster to the other
                for idx in cluster_list[len(distance_map) - closest_couple_index[0][1]].get_points():
                    cluster_list[len(distance_map) - closest_couple_index[0][0]].add_point(idx)

                # delete cluster
                cluster_list = np.delete(cluster_list, len(distance_map) - closest_couple_index[0][1], None)
                # delete rows
                distance_map = np.delete(distance_map, closest_couple_index[0][1], 0)
                # delete columns
                distance_map = np.delete(distance_map, 0, closest_couple_index[0][1])

                # update cluster distance
                new_distances = []

                for idx in idx_list:
                    new_distances.append(cluster_list[0].calculate_complete_linkage_criterion(df[idx]))

                for cluster in cluster_list:
                    new_distances.append(find_distance_between_clusters(cluster, cluster_list[len(cluster_list)]))

                distance_map[0][closest_couple_index[0][0]-1] = new_distances
                distance_map[closest_couple_index[0][0]-1][0] = new_distances

    return idx_list, distance_map, cluster_list


def main():

    # import data
    iris = load_iris()
    df = iris.data

    # initiate distance map
    distance_map = calculate_distance_map(df)
    print(distance_map)

    # set variables and start clustering
    cluster_list = []
    epoch = 3  # amount of clustering steps
    i = 0

    idx_list = np.arange(0, len(df), 1)
    while i < epoch:

        idx_list, distance_map, cluster_list = do_clustering(df, idx_list, distance_map, cluster_list)

        # generate map of colors
        color_condition = []
        color_counter = 0
        for index in np.arange(0, len(df), 1):
            # add color to single point
            if index in idx_list[:]:
                color_condition.append(color_counter)
                color_counter += 1
            # add color to point which is a part of a cluster
            else:
                for idx, cluster in enumerate(cluster_list):
                    if index in cluster.get_points():
                        color_condition.append(len(idx_list) + idx)
                        break

        figure, axis = plt.subplots(1, 1)

        plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=color_condition, cmap='brg')
        axis.set_xlabel('sepal length (cm)')
        axis.set_ylabel('sepal length (cm)')

        plt.show()

        i += 1


if __name__ == '__main__':
    main()
