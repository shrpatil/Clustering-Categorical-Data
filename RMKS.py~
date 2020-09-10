import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn import metrics
import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.testing import ignore_warnings
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster._dbscan_inner import dbscan_inner


# The dataset contains Categorical Features. These categorical features are modified to binary features
def data_modify():

    # loading abbreviations file
    # the file is encoded in latin-1
    raw_abbrs = [line.decode('latin-1') for line in open('stateabbr.txt', 'rb')]

    # get raw samples with the locations as a list
    raw_samples = [sample.decode('latin-1').strip().split(',') for sample in open('plants.data', 'rb')]

    # parse the abbreviations, every abbreviation is lower case
    samples_dict = {}
    sample_count = len(raw_samples)
    for line in raw_abbrs:
        m = re.search('^([a-z]{2,})\s', line)
        if m:
            abbr = m.group(0).replace(' ', '')
            # filled with zeros
            samples_dict[abbr] = np.zeros(sample_count, dtype=np.int8)
    print("Num Locations: {}".format(len(samples_dict.keys())))

    # there should be 70 locations, turns out "Prince Edward Island" does not have an abbreviation
    # also, samples have an abbreviation the location "pe" and "gl" which is not in stateabbr.txt
    # let's assume pe == "Prince Edward Island" and gl == "GreenLand" and add it to the dictionary
    samples_dict['pe'] = np.zeros(sample_count, dtype=np.int8)
    samples_dict['gl'] = np.zeros(sample_count, dtype=np.int8)

    print("Num Locations: {}".format(len(samples_dict.keys())))

    all_locations = set(samples_dict.keys())
    # assign 1 for the locations
    for i in range(0, sample_count):
        abbrs = set(raw_samples[i][1:])
        for abbr in abbrs:
            samples_dict[abbr][i] = 1

    plants_df = pd.DataFrame(data=samples_dict)
    number_of_clusters = [10, 20, 30, 40]
    number_of_dimensions = 2

    plants_df.to_csv('plants_modified.csv')
    print(len(plants_df))

    return plants_df


def kmeans_of_sample(df, sample_size, i, clusters):

    number_of_clusters = [10, 20, 30, 40]
    number_of_dimensions = 2
    temp_df = df.sample(sample_size)
    #temp_df = df.sample(100)
    pca = PCA(n_components=number_of_dimensions)
    pca.fit(temp_df)
    plants_2d = pca.transform(temp_df)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    fig = plt.figure(figsize=(20, 4))
    for i, clusters in enumerate(number_of_clusters):
        print("{} :: START :: i={} c={}".format(datetime.datetime.now(), i, clusters))
        tstart = datetime.datetime.now()
        fig.add_subplot(101 + i + 10 * len(number_of_clusters))
        kmeans = KMeans(n_clusters=clusters)
        l = kmeans.fit(temp_df)
        labels = ['cluster ' + str(label + 1) for label in kmeans.labels_]
        ax = sns.swarmplot(x=plants_2d[:, 0], y=plants_2d[:, 1], hue=labels)
        ax.set(xticklabels=[])
        ax.set(yticklabels=[])
        ax.legend(loc='upper right')
        if i == 3:
            ax.legend_.remove()


        print("{} :: DONE :: {}".format(datetime.datetime.now(), datetime.datetime.now() - tstart))
        silhouette_avg = silhouette_score(temp_df, kmeans.labels_)
        print("For n_clusters =", clusters, "The average silhouette_score is :", silhouette_avg)
        sample_silhouette_values = silhouette_samples(temp_df, kmeans.labels_)

    y_lower = 10
    X = temp_df.as_matrix()

    print("{} :: START :: i={} c={}".format(datetime.datetime.now(), i, clusters))
    tstart = datetime.datetime.now()
    fig.add_subplot(101 + i + 10 * len(number_of_clusters))
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(temp_df)
    cluster_labels = kmeans.labels_
    #print(cluster_labels)
    labels = ['cluster ' + str(label + 1) for label in kmeans.labels_]
    ax2 = sns.swarmplot(x=plants_2d[:, 0], y=plants_2d[:, 1], hue=labels, ax=ax2)
    ax2.set(xticklabels=[])
    ax2.set(yticklabels=[])
    ax2.legend(loc='upper right')
    if i == 3:
        ax2.legend_.remove()

    plt.show()
    print("{} :: DONE KMeans :: {}".format(datetime.datetime.now(), datetime.datetime.now() - tstart))
    silhouette_avg = silhouette_score(temp_df, kmeans.labels_)
    print("For n_clusters =", clusters, "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(temp_df, cluster_labels)

    for j in range(clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == j]

    return plants_2d


def apply_RMKS (df):

    tstart = datetime.datetime.now()
    initial_cluster_number=2
    # creating an object from RMKS class
    db = RMKS(eps=0.2, min_samples=2).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(df, labels))

    # Plotting the results
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = df[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = df[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    print("{} :: DONE :: {}".format(datetime.datetime.now(), datetime.datetime.now() - tstart))

    return

#scanning the attributes
def scan(X, eps=0.5, min_samples=5, metric='minkowski', metric_params=None,
           algorithm='auto', leaf_size=30, p=2, sample_weight=None,
           n_jobs=None):
    if not eps > 0.0:
        raise ValueError("eps must be positive.")

    X = check_array(X, accept_sparse='csr')
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        check_consistent_length(X, sample_weight)

    if metric == 'precomputed' and sparse.issparse(X):
        neighborhoods = np.empty(X.shape[0], dtype=object)
        X.sum_duplicates()  # XXX: modifies X's internals in-place

        # set the diagonal to explicit values, as a point is its own neighbor
        with ignore_warnings():
            X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

        X_mask = X.data <= eps
        masked_indices = X.indices.astype(np.intp, copy=False)[X_mask]
        masked_indptr = np.concatenate(([0], np.cumsum(X_mask)))
        masked_indptr = masked_indptr[X.indptr[1:-1]]

        # split into rows
        neighborhoods[:] = np.split(masked_indices, masked_indptr)
    else:
        neighbors_model = NearestNeighbors(radius=eps, algorithm=algorithm,
                                           leaf_size=leaf_size,
                                           metric=metric,
                                           metric_params=metric_params, p=p,
                                           n_jobs=n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, eps,
                                                         return_distance=False)

    if sample_weight is None:
        n_neighbors = np.array([len(neighbors)
                                for neighbors in neighborhoods])
    else:
        n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                for neighbors in neighborhoods])

        # Initially, all samples are noise.
    labels = np.full(X.shape[0], -1, dtype=np.intp)

    # A list of all core samples found.
    core_samples = np.asarray(n_neighbors >= min_samples, dtype=np.uint8)
    dbscan_inner(core_samples, neighborhoods, labels)
    return np.where(core_samples)[0], labels


# Implementing KMeans with a neighbouthood treshoold.
# The record points which exceeds the treshold distance is accepted as noise.
class RMKS(BaseEstimator, ClusterMixin):

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs



    def fit(self, X, y=None, sample_weight=None):

        X = check_array(X, accept_sparse='csr')
        clust = scan(X, sample_weight=sample_weight,
                       **self.get_params())
        self.core_sample_indices_, self.labels_ = clust
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self


def main():

    plants_df = data_modify()
    #print(plants_df)

    # This function executes regular kmeans
    #applied_PCA = apply_PCA(plants_df, 2, 200)
    for i, clusters in enumerate([2]):
        temp_df=kmeans_of_sample(plants_df, 1000, i, clusters)

    apply_RMKS(temp_df)



    return


main()
