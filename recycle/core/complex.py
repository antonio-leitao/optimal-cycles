import gudhi
import itertools
import numpy as np
from scipy.linalg import null_space


def create_simplicial_complex(points, max_edge_length, dimension, alpha_complex=False):
    if alpha_complex:
        return gudhi.AlphaComplex(points=points).create_simplex_tree(dimension)
    return gudhi.RipsComplex(
        points=points, max_edge_length=max_edge_length
    ).create_simplex_tree(max_dimension=dimension)


def get_representative_and_complex(gudhi_complex, dimension=2):
    """
    Manually reduces boundary matrix.
    Required to get reduced matrix and initial representative.
    Might be necessary to port it to lower level stuff like fortran.
    """
    boundary_map = {
        tuple(sorted(simplex)): set(
            itertools.combinations(tuple(sorted(simplex)), len(simplex) - 1)
        )
        - {tuple()}
        for simplex, _radius in gudhi_complex.get_filtration()
    }

    R = {k: v for k, v in boundary_map.items()}
    # V = { k : {k} for k in boundary_map}
    lowinv = {}  # lowinv[i]=index of column with the lowest 1 at i
    order_function = lambda s: (gudhi_complex.filtration(s), s)
    for s in sorted(R, key=order_function):
        t = lowinv.get(max(R[s], key=order_function), -1) if len(R[s]) != 0 else -1
        while t != -1:
            R[s] = R[t] ^ R[s]  # symmetric difference of t-th and s-th columns
            # V[s] = V[t]^V[s]
            t = lowinv.get(max(R[s], key=order_function), -1) if len(R[s]) != 0 else -1
        if len(R[s]) != 0:
            lowinv[max(R[s], key=order_function)] = s

    loops = [bar for bar in lowinv.items() if len(bar[0]) == 2]
    longest_loop = max(
        loops,
        key=lambda bar: gudhi_complex.filtration(bar[1])
        - gudhi_complex.filtration(bar[0]),
    )
    # Get representatives
    representative = R[longest_loop[1]]
    # representative = V[longest_loop[0]]
    # Get an homology base
    reduced_boundary_base = {
        simplex
        for simplex, boundary in R.items()
        if len(simplex) == dimension + 1 and len(boundary) > 0
    }
    return representative, boundary_map, reduced_boundary_base, longest_loop


class BoundaryNotPresentError(Exception):
    """Custom exception for when a boundary is not found in n_tuples."""

    pass


def build_boundary_matrix(n_tuples, n_plus_1_tuples):
    """
    Builds a boundary matrix from two lists of tuples.

    Args:
        n_tuples: List of tuples of length n
        n_plus_1_tuples: List of tuples of length n+1

    Returns:
        List[List[int]]: Matrix where rows correspond to n_tuples and columns to n_plus_1_tuples

    Raises:
        ValueError: If input tuples have invalid lengths
        BoundaryNotPresentError: If a boundary is not found in n_tuples
    """
    # Verify input lengths
    if not n_tuples or not n_plus_1_tuples:
        return []

    n = len(n_tuples[0])
    if any(len(t) != n for t in n_tuples) or any(
        len(t) != n + 1 for t in n_plus_1_tuples
    ):
        raise ValueError("Invalid tuple lengths")

    # Initialize matrix with zeros
    matrix = [[0 for _ in range(len(n_plus_1_tuples))] for _ in range(len(n_tuples))]

    # Build matrix column by column
    for col, n_plus_1_tuple in enumerate(n_plus_1_tuples):
        # Generate all possible boundaries (n-tuples) from the (n+1)-tuple
        for i in range(n + 1):
            # Create boundary by removing element at position i
            boundary = n_plus_1_tuple[:i] + n_plus_1_tuple[i + 1 :]

            try:
                row = n_tuples.index(boundary)
            except ValueError:
                raise BoundaryNotPresentError(
                    f"Boundary {boundary} from {n_plus_1_tuple} not found in n_tuples"
                )

            # Calculate coefficient and assign to matrix
            coefficient = (-1) ** (i + 1)
            matrix[row][col] = coefficient

    return np.array(matrix)


def lift_representative(representative, nm1_simplices, n_simplices):
    partial_nm1 = build_boundary_matrix(nm1_simplices, n_simplices)
    basis_index = np.sort([n_simplices.index(simp) for simp in representative])

    A = partial_nm1[:, basis_index]
    # Compute the null space of A
    null_space_A = null_space(A)
    nonzero_entries = np.count_nonzero(~np.isclose(null_space_A, 0), axis=0)
    sparse_column = null_space_A[:, np.argmax(nonzero_entries)]
    # Here we will simply take the sign and convert it to 1s and -1s
    lifted_vector = sparse_column / np.max(sparse_column)
    x_orig = np.zeros(
        shape=partial_nm1.shape[1]
    )  # zeros like the columns of boundary_matrix
    x_orig[basis_index] = lifted_vector

    assert np.all(np.isclose(partial_nm1.dot(x_orig), 0))

    return x_orig
