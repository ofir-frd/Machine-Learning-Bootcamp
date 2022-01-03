import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Dbscan:

    def __init__(self, new_data: np.ndarray, new_epsilon: float = 0.1, neighbours_limit: int = 3, core_idx_list=None,
                 done_idx_list=None):
        if done_idx_list is None:
            done_idx_list = []
        if core_idx_list is None:
            core_idx_list = []
        self.points: np.ndarray = new_data
        # 2-D index matrix with each row containing indexes of each unique cluster.
        self.core_idx_array: np.ndarray = core_idx_list
        self.done_idx: np.ndarray = done_idx_list
        self.epsilon: float = new_epsilon
        self.neighbours_for_core: int = neighbours_limit

    def region_query(self, point_idx):
        """ Calculate the amount of points in distance epsilon to reference point
        # Arguments
            point_inx: index of reference point in the points array
        # Output
            int: amount of neighbours to the reference point
        """
        neighbours_counter = 0
        for point in self.points:
            if 0 < np.sum((point[:] - self.points[point_idx][:])**2) <= self.epsilon:
                neighbours_counter += 1

        return neighbours_counter

    def expand_cluster(self, cluster_seed_point_idx: int):
        """ Expend a cluster starting for a random core point.
            Will continue to search for core and neighbours until exhausted.
            The points index in the cluster will be added to a new vector of  self.core_inx_array
        and flagged as done in the self.done_inx list
         # Arguments
            available_point_idx: index of reference point in the points array
        """

        if len(self.done_idx) == len(self.points):
            return

        for idx, point in enumerate(self.points):

            # skip point if it was already analysed
            if idx in self.done_idx:
                continue

            # check if point in epsilon of core
            if np.sum((point[:] - self.points[cluster_seed_point_idx]) ** 2) <= self.epsilon:
                self.core_idx_array[-1].append(idx)
                self.done_idx.append(idx)

                self.plot_clusters()
                
                # calculate amount of neighbours
                neighbours_counter = self.region_query(idx)

                # Check if new cluster member point can be a core, if yes, expand
                if neighbours_counter >= self.neighbours_for_core:
                    self.expand_cluster(idx)               
                    

    def fit(self):
        """ Iterates over all points, finds core points and send them to expand_cluster function
        to expend.
        """

        if len(self.done_idx) == len(self.points):
            print('DBSCAN completed...')
            return

        for ind in range(0, len(self.points), 1):

            # skip point if it was already analysed
            if ind in self.done_idx:
                continue

            # calculate amount of neighbours
            neighbours_counter = self.region_query(ind)

            # Check if point can be a core, if yes start a cluster
            if neighbours_counter >= self.neighbours_for_core:
                self.core_idx_array.append([ind])
                self.done_idx.append(ind)
                self.expand_cluster(ind)
                print('Completed the expedition of cluster: ', (len(self.core_idx_array) - 1))
                
            
            
        print('DBSCAN completed...')

    def plot_clusters(self):
        """ Color code each point per cluster and plot
        """
        color_condition = np.zeros(len(self.points))

        for core_number, core_indexes in enumerate(self.core_idx_array):
            color_condition[core_indexes] = core_number+1

        # plot
        figure, axis = plt.subplots(1, 1)

        plt.scatter(self.points[:, 0], self.points[:, 1], c=color_condition)
        axis.set_xlabel('sepal length (cm)')
        axis.set_ylabel('sepal length (cm)')

        plt.show(block=False)


def main():
    # import some data to play with
    iris = datasets.load_iris()
    x_train = iris.data[:, [1, 2]]  # we only take the first two features.

    new_epsilon = 0.7
    neighbours_limit = 3
    dbscan_obj = Dbscan(x_train, new_epsilon, neighbours_limit)

    dbscan_obj.fit()

    dbscan_obj.plot_clusters()


if __name__ == '__main__':
    main()
