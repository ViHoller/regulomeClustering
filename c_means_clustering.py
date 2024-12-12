import igraph as ig
import pandas as pd
import numpy as np
from compress_pickle import load, dump
from statistics import mean
from IPython.display import clear_output
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy


### Distances
# dist_sum distance measure calculation
# 1 / sum of w of edges from node to cluster
def inverse_sum_PPV_dist(neighbors, cluster):
    std_dist = 1000
    
    # get weights of edges of node to neighbors in cluster
    weights_sum = sum([neighbors[node] for node in neighbors.keys() if node in cluster])
    if weights_sum == 0: # check if there is any edge
        return std_dist # return standard value
    
    return 1 / weights_sum


# path_len distance measure calculation
# distance by average path length to all nodes in cluster
def path_length_dist(node, path_lengths, cluster):   
    return mean([path_lengths[cluster_node] for cluster_node in cluster if cluster_node != node])


# edge_ratio distance meaesure (preferred)
# distance of weights to the of edge to neighbors in the cluster
# normalized by the degree of the node
def edge_ratio(neighbors, cluster):
    weights = [neighbors[node] for node in neighbors.keys() if node in cluster]
    if len(weights) == 0:
        return 1
    weights_mean = mean(weights) # mean needs at least one point, so have to check before
    node_edge_ratio = len(weights) / len(neighbors)
    return 1 - node_edge_ratio * weights_mean


distance_measures = {
    'dist_sum': inverse_sum_PPV_dist,
    'edge_ratio': edge_ratio,
    'path_len': path_length_dist
}


# Funtion to calculcate all distances from node to clusters
# Handles which distance measure to use
def node_c_distances(node, neighbors, clusters, distance_measure): # add weight formula argument:
    node_distances = dict()

    for (cluster_id, cluster) in clusters.items():
        try:
            node_distances[cluster_id] = distance_measures[distance_measure](neighbors, cluster)
        except KeyError:
            print("Somehow you got till here with a wrong distance measure, weird")
            return 
    return node_distances


### Function to calculate membership values for node
def calc_membership(node_distances, m):
    node_memberships = dict()
    memberships_heuristic = dict()
    fuzzy_exponent = (1 / (m - 1))

    for (cluster_id, d) in node_distances.items():         
        try: 
            node_memberships[cluster_id] = memberships_heuristic[d]
        except KeyError:
            # give better name - denomintation or membership function  
            membership_den = [(d / other_d) ** fuzzy_exponent for other_d in node_distances.values()]
            membership = 1 / sum(membership_den)

            node_memberships[cluster_id] = membership
            memberships_heuristic[d] = membership

    del memberships_heuristic
    return node_memberships

### Function to calcualte node addition to objective function J (optimization function)
def optimization_function(node_distances, node_memberships, m): # change function name
    node_J = 0
    for cluster_id in node_distances.keys():
        node_J += node_memberships[cluster_id]**m * node_distances[cluster_id]**2
    return node_J
    

### Function for iteration of each node, calculates distance, membership and J
# Necessary for parallelization
def node_iteration(node, neighbors, clusters, m, optimize, distance_measure):
    node_distances = node_c_distances(node, neighbors, clusters, distance_measure) # returns dictionary with distance to each cluster
    node_memberships = calc_membership(node_distances, m) # adds memberships of node to dictionary
    if optimize:
        Ji = optimization_function(node_distances, node_memberships, m)
        return node, node_distances, node_memberships, Ji
    return node, node_distances, node_memberships


# generate neighbor dictionary
def gather_neighbors(graph, nodes):
    neighbors_dict = {
        node : {
            graph.vs[neighbor]['name'] : graph.es[graph.get_eid(node, neighbor)]['PPV'] for neighbor in graph.neighbors(node)
            } 
            for node in nodes
        }
    return neighbors_dict


def update_clusters_perc(memberships_dict, clusters, percentile):
    print(f"Updating Clusters: percentile {percentile}")
    
    for (node, memberships) in memberships_dict.items():
        perc_lim = np.percentile(list(memberships.values()), percentile)
        node_clusters = [cluster_id for (cluster_id, membership) in memberships.items() if membership > perc_lim]    
        for cluster_id in node_clusters:
            clusters[cluster_id].add(node)
    
    return clusters


def update_clusters_thresh(memberships_dict, clusters, t):
    for (node, memberships) in memberships_dict.items():
        node_clusters = [cluster_id for (cluster_id, membership) in memberships.items() if membership > t]
        for cluster_id in node_clusters:
            clusters[cluster_id].add(node)
    return clusters


### Main function
# path_lengths only necessarz when using the path_len distance measure
def network_c_means(graph, clusters, m, n_iter, optimize=False, percentile=0.95, t = 0.5, distance_measure='dist_sum', path_lengths=None, cores=6):
    # add leiden here after 
    nodes = [node['name'] for node in graph.vs]
    Js = list()
    # clusters = {cluster_id:cluster for (cluster_id, cluster) in clusters.items() if len(cluster) >= 20}
    
    if distance_measure == 'path_len':
        if path_lengths == None:
            print("Must give dictionary containing path lengths using path_lengths argument")
            return
        connections_dict = path_lengths # maybe change name, cuase connections_dict kinda silly
        del path_lengths
    else:
        connections_dict = gather_neighbors(graph, nodes)

    del graph

    parallel = Parallel(n_jobs=cores, verbose=0, batch_size=32)
    cluster_history = [deepcopy(clusters)]

    for i in range(1,n_iter+1):
        print(f'Iteration {i} of {n_iter}')

        results = parallel(delayed(node_iteration)(node, connections_dict[node], clusters, m, optimize, distance_measure) for node in tqdm(nodes))
        
        if optimize:
            Js.append(sum(result[3] for result in results)) # maybe make results into a named tuple
        
        memberships_dict = {result[0] : result[2] for result in results}
        # distances_dict = {result[0] : result[1] for result in results}
        assert type(memberships_dict) == dict
        ### MAYBE DO THIS ALREADY IN THE NODE ITERATION FUNCTION, AND THEN 
        clusters = update_clusters_perc(memberships_dict, clusters, percentile=percentile)
        # clusters = update_clusters_thresh(memberships_dict, clusters, t)

        clusters = {cluster_id : cluster for cluster_id, cluster in clusters.items() if len(cluster) < 3000}
        
        cluster_history.append(deepcopy(clusters))

        
    
    return cluster_history, Js