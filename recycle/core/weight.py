import numpy as np


def create_weight_matrix(inv_row_map, weight_function) -> np.ndarray:
    n = len(inv_row_map)
    weight_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            simplex_i = inv_row_map[i]
            simplex_j = inv_row_map[j]
            weight_matrix[i, j] = weight_function(simplex_i, simplex_j)

    return weight_matrix


def edge_length_function(points):
    def edge_length(simp1, simp2):
        if simp1 == simp2:
            return np.linalg.norm(points[simp1[0]] - points[simp1[1]])
        else:
            return 0

    return edge_length


def time_adjacency_matrix():
    def inner_time_adjacency_matrix(simplex_j, simplex_i):
        if are_simplices_adjacent(simplex_j, simplex_i):
            return time_weight(simplex_i, simplex_j)
        else:
            return 0  # <------------THINK ABOUT THIS

    return inner_time_adjacency_matrix


def time_label(simplex_i):
    return np.mean(simplex_i)


def time_weight(simp1, simp2) -> float:
    return np.abs(time_label(simp1) - time_label(simp2))


def are_simplices_adjacent(simplex1, simplex2):
    # Count common elements between the two tuples
    common_elements = set(simplex1).intersection(set(simplex2))
    # Two simplices are adjacent if they share exactly n-1 elements
    if len(common_elements) == len(simplex1) - 1:
        return 1
    return 0
