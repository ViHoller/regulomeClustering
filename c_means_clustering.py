import igraph as ig
import pandas as pd
import numpy as np
from compress_pickle import load, dump
from statistics import mean
from IPython.display import clear_output
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy


def inverse_sum_PPV_dist(neighbors, cluster):
    std_dist = 1000
    
    connected_nodes = cluster.intersection(set(neighbors.keys()))
    if len(connected_nodes) == 0:
        return std_dist # determine the standard distance - 1.5/min(weight) 
    
    weights_sum = sum([neighbors[node] for node in connected_nodes])
    if weights_sum == 0:
        return std_dist
    
    return 1 / weights_sum


def path_length_dist(node, path_lengths, cluster):   
    return mean([path_lengths[cluster_node] for cluster_node in cluster if cluster_node != node])


def edge_ratio(neighbors, cluster):
    
    connected_nodes = cluster.intersection(set(neighbors.keys()))
    if len(connected_nodes) == 0:
        return 1 
    
    weights_mean = mean([neighbors[node] for node in connected_nodes])
    node_edge_ratio = len(connected_nodes) / len(neighbors)
    return 1 - node_edge_ratio * weights_mean

def node_c_distances(node, neighbors, clusters, distance_measure): # add weight formula argument:
    node_distances = dict()

    for (cluster_id, cluster) in clusters.items():
        match distance_measure:
            case 'dist_sum':
                node_distances[cluster_id] =  inverse_sum_PPV_dist(neighbors, cluster)# set 1.5 as max distance, should I set other to more ?
            case 'path_len':
                node_distances[cluster_id] = path_length_dist(node, neighbors, cluster)
            case 'edge_ratio':
                node_distances[cluster_id] = edge_ratio(neighbors, cluster)
            case _:
                return # quit function if wrong function name
    return node_distances


### Function to calculate membership values for node
def calc_membership(node_distances, m):
    node_memberships = dict()

    for (cluster_id, d) in node_distances.items():         
        fuzzy_exponent = (1 / (m - 1))

        # give better name - denomintation or membership function  
        membership_den = [(d / other_d) ** fuzzy_exponent for other_d in node_distances.values()]
        node_memberships[cluster_id] = 1 / sum(membership_den)

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


def gather_neighbors(graph, nodes, distance_measure='dist_sum'):
    match distance_measure:
        case 'dist_sum':
            return {
                node : {
                    graph.vs[neighbor]['name'] : graph.es[graph.get_eid(node, neighbor)]['PPV'] for neighbor in graph.neighbors(node)
                    } 
                    for node in nodes
                }
        case 'edge_ratio':
            return {
                node : {
                    graph.vs[neighbor]['name'] : graph.es[graph.get_eid(node, neighbor)]['PPV'] for neighbor in graph.neighbors(node)
                    } 
                    for node in nodes
                }
        case _:
            print("UNrecognized distance measure")
            return 


def update_clusters_perc(memberships_dict, clusters, percentile):
    print(f"Updating Clusters: percentile {percentile}")
    for (node, memberships) in memberships_dict.items():
        perc_lim = np.percentile(list(memberships.values()), percentile)
        # print(perc_lim, min(memberships.values()))
        node_clusters = [cluster_id for (cluster_id, membership) in memberships.items() if membership > perc_lim]
        # print(len(node_clusters))
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
    memberships_dict = dict()
    # clusters = {cluster_id:cluster for (cluster_id, cluster) in clusters.items() if len(cluster) >= 20}
    
    if distance_measure == 'path_len':
        if path_lengths == None:
            print("Must give dictionary containing path lengths using path_lengths argument")
            return
        connections_dict = path_lengths # maybe change name, cuase connections_dict kinda silly
        del path_lengths
    else:
        connections_dict = gather_neighbors(graph, nodes, distance_measure)
    if optimize:
        Js = list()

    del graph

    parallel = Parallel(n_jobs=cores, verbose=0, batch_size=32)
    cluster_history = [deepcopy(clusters)]

    for i in range(1,n_iter+1):
        print(f'Iteration {i} of {n_iter}')

        results = parallel(delayed(node_iteration)(node, connections_dict[node], clusters, m, optimize, distance_measure) for node in tqdm(nodes))
        Js.append(sum(result[3] for result in results)) # maybe make results into a named tuple
        
        # return {result[0] : result[2] for result in results}
        memberships_dict = {result[0] : result[2] for result in results}
        distances_dict = {result[0] : result[1] for result in results}

        clusters = update_clusters_perc(memberships_dict, clusters, percentile=percentile) 
        # clusters = update_clusters_thresh(memberships_dict, clusters, t)
        # clusters = {cluster_id:cluster for (cluster_id, cluster) in clusters.items() if len(cluster) <= 2500}

        cluster_history.append(deepcopy(clusters))
        

        # del results, memberships_dict
        # clear_output()
    return cluster_history, Js, memberships_dict, distances_dict