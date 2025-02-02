import time
import random
import numpy as np
from geomstats.hypersphere import HypersphereMetric
from .karcher_mean import karcher_mean

# K-Means Algorithm using Euclidean mean computation
def kmeans(k, datapoints):
    # d - Dimensionality of Datapoints
    N, d = datapoints[0].shape
    dim = N * d
    metric = HypersphereMetric(dimension=dim - 1)
    d = N
    # Limit our iterations
    Max_Iterations = 1000
    i = 0

    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)

    # Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0, k):
        cluster_centers += [random.choice(datapoints)]

        # Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        # In this particular implementation we want to force K exact clusters.
        # To take this feature off, simply take away "force_recalculation" from the while conditional.
        force_recalculation = False

    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation):

        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1

        # Update Point's Cluster Alligiance
        for p in range(0, len(datapoints)):
            min_dist = float("inf")

            # Check min_distance against all centers
            for c in range(0, len(cluster_centers)):
                d1 = np.asarray(datapoints[p]).reshape(1, dim)
                d2 = np.asarray(cluster_centers[c]).reshape(1, dim)
                dist = metric.dist(d1, d2)

                if (dist < min_dist):
                    min_dist = dist
                    cluster[p] = c  # Reassign Point to new Cluster

        # Update Cluster's Position
        for k in range(len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(len(datapoints)):
                if cluster[p] == k:  # If this point belongs to the cluster
                    for j in range(0, d):
                        new_center[j] += datapoints[p][j]
                    members += 1

            for j in range(0, d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)

                    # This means that our initial random assignment was poorly chosen
                # Change it to a new datapoint to actually force k clusters
                else:
                    new_center = random.choice(datapoints)
                    force_recalculation = True
                    print("Forced Recalculation...")

            cluster_centers[k] = new_center

    print("======== Results ========")
    print("Iterations ", i)
    print("Assignments ", cluster)

    return cluster_centers


# K-Means Algorithm using Karcher mean computation
def kmeans_kendall(k, datapoints):
    # d - Dimensionality of Datapoints
    N, d = datapoints[0].shape
    dim = N * d
    metric = HypersphereMetric(dimension=dim - 1)
    d = N
    # Limit our iterations
    max_iterations = 1000

    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)

    # Randomly Choose Centers for the Clusters
    cluster_centers = []
    for i in range(0, k):
        cluster_centers += [random.choice(datapoints)]

        # Sometimes The Random points are chosen poorly and so there ends up being empty clusters
        # In this particular implementation we want to force K exact clusters.
        # To take this feature off, simply take away "force_recalculation" from the while conditional.
        force_recalculation = False
    i = 0
    while (cluster != prev_cluster) or (i > max_iterations) or force_recalculation:
        print('iter %d' % i)
        start = time.time()
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1

        # Update Point's Cluster Alligiance
        for p in range(0, len(datapoints)):
            min_dist = float("inf")

            # Check min_distance against all centers
            for c in range(0, len(cluster_centers)):
                d1 = np.asarray(datapoints[p]).reshape(1, dim)
                d2 = np.asarray(cluster_centers[c]).reshape(1, dim)
                dist = metric.dist(d1, d2)

                if dist < min_dist:
                    min_dist = dist
                    cluster[p] = c  # Reassign Point to new Cluster

        # Update Cluster's Position
        for k in range(len(cluster_centers)):
            cluster_members = []
            for p in range(len(datapoints)):
                if cluster[p] == k:  # If this point belongs to the cluster
                    cluster_members.append(datapoints[p])

            if len(cluster_members) > 0:
                new_center = karcher_mean(cluster_members)

                # This means that our initial random assignment was poorly chosen
            # Change it to a new datapoint to actually force k clusters
            else:
                new_center = random.choice(datapoints)
                force_recalculation = True
                print("Forced Recalculation...")

            cluster_centers[k] = new_center
            print(time.time() - start)

    print("======== Results ========")
    print("Iterations ", i)
    print("Assignments ", cluster)

    return cluster_centers


# TESTING THE PROGRAM#
if __name__ == "__main__":
    # 2D - Datapoints List of n d-dimensional vectors. (For this example I already set up 2D Tuples)
    # Feel free to change to whatever size tuples you want...
    datapoints = [(3, 2), (2, 2), (1, 2), (0, 1), (1, 0), (1, 1), (5, 6), (7, 7), (9, 10), (11, 13), (12, 12), (12, 13),
                  (13, 13)]

    k = 2  # K - Number of Clusters

    kmeans(k, datapoints)