import numpy as np
import load_data as ld
import matplotlib.pyplot as plt
import itertools



class K_Means():
    ''' encapsulates k means optimizer'''
    def __init__(self, k=5, initialization='uniform'):
        assert initialization in ['uniform', 'by_distance']
        self.k = k
        self.X = None
        self.initialization = initialization
        self.fitted = False
        self.centroids = None  # 2D-array, (k x data_dimension)
        self.assignment = None # one cluster index per data item

    @property
    def data_dimension(self):
        assert self.X is not None, "data not initialized"
        return self.X.shape[1]

    @property
    def number_data(self):
        assert self.X is not None, "data not initialized"
        return self.X.shape[0]


    def fit(self, X, return_details=False, plot=False):
        self.X = X
        self.assignment = np.zeros(self.number_data)
        # initialize clusters
        self.centroids = np.zeros((0, self.data_dimension))
        if self.initialization == 'uniform':
            c_indices = np.array(np.random.choice(range(self.number_data), size=self.k))
            self.centroids = self.X[c_indices]
        else:
            for i in range(self.k):
                p = self.get_weights()
                new_idx = np.random.choice(range(self.number_data), size=1, p=p)
                new_centroid = self.X[new_idx]
                self.centroids = np.concatenate((self.centroids, new_centroid))
        #assert list(self.centroids) == list(set(list(self.centroids))), "There shouldnt be any duplicate centroids!"

        # iterate k-means until convergence
        has_converged = False
        steps_for_convergence = -1
        for i in range(10000):
            old_assignment = self.assignment.copy()
            for j, point in enumerate(self.X):
                self.assignment[j] = self.closest_centroid(point, return_centroid_index=True)
            self.reassign_clusters()
            # determine convergence
            if np.all(self.assignment == old_assignment):
                steps_for_convergence = i # (convergence reached one step before the current step)
                has_converged = True
                break
        # return clusters
        self.fitted = True
        if return_details:
            return has_converged, steps_for_convergence

    def mean_centroid_distance(self):
        assert self.fitted
        centroid_matrix = np.array([self.centroids[c] for c in self.assignment])
        assert centroid_matrix.shape == self.X.shape
        return np.mean((centroid_matrix - self.X)**2)

    def classify(self, X2):
        ''' Assigns closest centroids to points in X2.
            :returns an array of cluster indices as long as X2.'''
        assert self.fitted
        assert X2.shape[1] == self.data_dimension
        X2_assignment = np.zeros(len(X2))
        for i, point in enumerate(X2):
            X2_assignment[i] = self.closest_centroid(point, return_centroid_index=True)
        return X2_assignment


    def reassign_clusters(self):
        assert self.assignment is not None
        assert self.X is not None
        assert len(self.assignment) == len(self.X)
        for c in range(self.k):
            X_c = self.X[self.assignment == c]
            self.centroids[c] = np.mean(X_c, axis=0)

    def get_weights(self):
        ''':return: 1D array of weights, one for each datapoint. Weights sum to one.'''
        assert self.X is not None
        if self.initialization == 'uniform' or self.centroids is None or len(self.centroids) == 0:
            return np.array([1./len(self.X)]*len(self.X))
        else:
            weights_unnorm = np.array([self.sqared_distance_closest_centroid(x) for x in self.X])
            W = np.sum(weights_unnorm)
            weights = weights_unnorm / W
            return weights

    def closest_centroid(self, point, return_centroid_index=False):
        '''  :param point: a 1D list or np-array  '''
        assert self.centroids is not None
        assert len(point) == self.data_dimension
        dists = np.sum((np.array(point)[np.newaxis, :] - self.centroids)**2, axis=1)
        cent_index = np.argmin(dists)
        if return_centroid_index:
            return cent_index
        else:
            return self.centroids[cent_index]


    def sqared_distance_closest_centroid(self, point):
        '''  :param point: a list or 1D np-array  '''
        assert self.centroids is not None
        assert len(point) == self.data_dimension
        dists = np.sum((np.array(point)[np.newaxis, :] - self.centroids)**2, axis=1)
        return np.min(dists)

# class K_Means end
###############################################################

def find_best_fitting_labelling(assignments, labels, return_loss=True):
    ''' among all possible cluster-index to label assignments, find the one that results in highest accuracy for this data.
        :param assignments: list / array of cluster indices, shape (N)
        :param labels: list / array of "true" labels (also integer values), shape (N)
        '''
    label_vals = np.unique(labels)
    assignment_vals = np.unique(assignments)
    assert len(label_vals) == len(assignment_vals)
    all_permuts = list(itertools.permutations(label_vals))
    all_losses = []
    for permut in all_permuts:
        matching_dict = {key: val for key, val in zip(assignment_vals, permut)}
        matched_assignments = np.array([matching_dict[key] for key in assignments])
        loss = np.sum(matched_assignments != labels)
        all_losses.append(loss)
    best_idx = np.argmin(np.array(all_losses))
    best_loss = all_losses[best_idx]
    best_permut = all_permuts[best_idx]
    best_matching_dict = {key: val for key, val in zip(assignment_vals, best_permut)}
    if return_loss:
        return best_matching_dict, best_loss
    else:
        return best_matching_dict

###############################################################

def main():

    df_train, df_val = ld.load_iris_data()
    kmeans = K_Means(k=3,initialization='by_distance')
    has_converged, steps_for_convergence = kmeans.fit(df_train.X, return_details=True)
    print(kmeans.centroids)

    # run both methods a couple of times, get average steps for convergence and avg optimal loss for each
    # 1. Uniform init:
    n = 100
    uniform_losses = np.zeros(n)
    uniform_losses_val = np.zeros(n)
    uniform_steps = np.zeros(n)
    uniform_has_not_converged = np.zeros(n)
    for i in range(n):
        kmeans = K_Means(k=3,initialization='uniform')
        has_converged, steps = kmeans.fit(df_train.X, return_details=True)
        uniform_has_not_converged[i] = not has_converged
        uniform_steps[i] = steps
        assignment = kmeans.assignment
        assignment_val = kmeans.classify(df_val.X)
        _, uniform_losses[i]     = find_best_fitting_labelling(assignment,     df_train.y)
        _, uniform_losses_val[i] = find_best_fitting_labelling(assignment_val, df_val.y)



    # 2. k-means++:
    plusplus_losses = np.zeros(n)
    plusplus_losses_val = np.zeros(n)
    plusplus_steps = np.zeros(n)
    plusplus_has_not_converged = np.zeros(n)
    for i in range(n):
        kmeans = K_Means(k=3,initialization='by_distance')
        has_converged, steps = kmeans.fit(df_train.X, return_details=True)
        plusplus_has_not_converged[i] = not has_converged
        plusplus_steps[i] = steps
        assignment = kmeans.assignment
        assignment_val = kmeans.classify(df_val.X)
        _, plusplus_losses[i]     = find_best_fitting_labelling(assignment,     df_train.y)
        _, plusplus_losses_val[i] = find_best_fitting_labelling(assignment_val, df_val.y)

    # Comparison between both methods:
    print("\nmean number of convergence steps:")
    print("Uniform initialization: \t"+ str(np.mean(uniform_steps)))
    print("k-means++: \t"+ str(np.mean(plusplus_steps)))
    print("\nNumber of times not converged: ")
    print("Uniform initialization:\t"+ str(np.mean(uniform_has_not_converged)))
    print("k-means++: \t"+ str(np.mean(plusplus_has_not_converged)))
    print("\nMean squared centroid distance after convergence:")
    print("Uniform initialization: \t"+ str(np.mean(uniform_steps)))
    print("k-means++: \t"+ str(np.mean(plusplus_steps)))
    print("\n")
    print("\nAccuracy under best cluster assignment: ")
    print("Uniform initialization: \t"+ str(np.mean(uniform_steps)))
    print("k-means++: \t"+ str(np.mean(plusplus_steps)))




if __name__ == "__main__":

    main()