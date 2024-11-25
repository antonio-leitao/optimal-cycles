import numpy as np
import pulp


def find_minimal_homologous_cycle(
    boundary_matrix, x_orig, W=None, use_gurobi=True, verbose=True
):
    """
    Solve the weighted linear optimization problem for finding a minimal homologous cycle.

    Parameters:
    - boundary_matrix: The boundary matrix mapping (n+1)-simplices to n-simplices
    - x_orig: The original cycle of n-simplices
    - W: Weight matrix for the objective function (default: identity matrix)

    Returns:
    - x_solution: The minimal homologous cycle
    - status: The optimization status
    - objective_value: The weighted L1 norm of the minimal homologous cycle
    """
    # Dimensions
    (
        m,
        n,
    ) = boundary_matrix.shape  # m: number of n-simplices, n: number of (n+1)-simplices

    # Check and prepare the weight matrix

    if isinstance(W, np.ndarray) and W.shape != (m, m):
        raise ValueError(f"W must be a square matrix of size {m}x{m}")

    # Create the LP problem
    prob = pulp.LpProblem("Weighted_Minimal_Homologous_Cycle", pulp.LpMinimize)

    # Decision variables
    x_plus = [pulp.LpVariable(f"x_plus_{i}", lowBound=0) for i in range(m)]
    x_minus = [pulp.LpVariable(f"x_minus_{i}", lowBound=0) for i in range(m)]
    q = [
        pulp.LpVariable(f"q_{i}", cat="Continuous") for i in range(n)
    ]  # CHECK THE LOW BOUND HERE

    if W is None:
        # Objective function: minimize L1 norm of x
        prob += pulp.lpSum(x_plus) + pulp.lpSum(x_minus)
    else:
        # Objective function: minimize weighted L1 norm of x
        # prob += pulp.lpSum(W[i, j] * x_plus[j] for i in range(m) for j in range(m)) + \
        #        pulp.lpSum(W[i, j] * x_minus[j] for i in range(m) for j in range(m))
        prob += pulp.lpSum(
            W[i, j] * (x_plus[j] + x_minus[j]) for i in range(m) for j in range(m)
        )

    # Constraints
    for i in range(m):
        prob += x_plus[i] - x_minus[i] == x_orig[i] + pulp.lpSum(
            boundary_matrix[i][j] * q[j] for j in range(n)
        )

    # Solve the problem
    if use_gurobi:
        solver = pulp.getSolver("GUROBI_CMD", msg=verbose)
    else:
        solver = pulp.PULP_CBC_CMD(msg=verbose)  # Using CBC as a default solver
    prob.solve(solver)

    # Extract the solution
    x_solution = np.array(
        [pulp.value(x_plus[i]) - pulp.value(x_minus[i]) for i in range(m)]
    )

    return x_solution, prob.status, pulp.value(prob.objective)


def find_minimal_spread_cycle(boundary_matrix, x_orig, W, alpha=0.7):
    """
    Solve the weighted linear optimization problem for finding a minimal homologous cycle.

    Parameters:
    - boundary_matrix: The boundary matrix mapping (n+1)-simplices to n-simplices
    - x_orig: The original cycle of n-simplices
    - W: Weight matrix for the objective function (default: identity matrix)

    Returns:
    - x_solution: The minimal homologous cycle
    - status: The optimization status
    - objective_value: The spread of the weighted values
    """
    # Dimensions
    (
        m,
        n,
    ) = boundary_matrix.shape  # m: number of n-simplices, n: number of (n+1)-simplices

    # Check and prepare the weight matrix
    if isinstance(W, np.ndarray) and W.shape != (m, m):
        raise ValueError(f"W must be a square matrix of size {m}x{m}")

    # Create the LP problem
    prob = pulp.LpProblem("Weighted_Minimal_Homologous_Cycle", pulp.LpMinimize)

    # Decision variables
    x_plus = [pulp.LpVariable(f"x_plus_{i}", lowBound=0) for i in range(m)]
    x_minus = [pulp.LpVariable(f"x_minus_{i}", lowBound=0) for i in range(m)]
    q = [pulp.LpVariable(f"q_{i}", cat="Continuous") for i in range(n)]

    # Auxiliary variables for spread
    min_val = pulp.LpVariable("min_val", lowBound=0, cat="Continuous")
    max_val = pulp.LpVariable("max_val", lowBound=0, cat="Continuous")

    # Objective: minimize the spread (max - min)
    prob += alpha * (max_val - min_val) + (1 - alpha) * (
        pulp.lpSum(x_plus) + pulp.lpSum(x_minus)
    )

    # Constraints for defining min and max
    for i in range(m):
        # Compute the value of Wx for the decision variables
        wx_value = pulp.lpSum(W[i, j] * (x_plus[j] + x_minus[j]) for j in range(m))

        # Ensure min_val is less than or equal to every wx_value
        prob += wx_value >= min_val

        # Ensure max_val is greater than or equal to every wx_value
        prob += wx_value <= max_val

        # Homology constraints
        prob += x_plus[i] - x_minus[i] == x_orig[i] + pulp.lpSum(
            boundary_matrix[i][j] * q[j] for j in range(n)
        )

    # Solve the problem
    prob.solve()

    # Extract the solution
    x_solution = np.array(
        [pulp.value(x_plus[i]) - pulp.value(x_minus[i]) for i in range(m)]
    )

    return x_solution, prob.status, pulp.value(prob.objective)


def minimize_sum_of_max_spread(boundary_matrix, x_orig, W):
    """
    Solve the linear optimization problem for finding a minimal homologous cycle,
    minimizing the sum of maximum row sums of the weighted matrix.

    Parameters:
    - boundary_matrix: The boundary matrix mapping (n+1)-simplices to n-simplices
    - x_orig: The original cycle of n-simplices
    - W: Weight matrix for the objective function (default: identity matrix)

    Returns:
    - x_solution: The minimal homologous cycle
    - status: The optimization status
    - objective_value: The sum of maximum row sums
    """
    # Dimensions
    (
        m,
        n,
    ) = boundary_matrix.shape  # m: number of n-simplices, n: number of (n+1)-simplices

    # Check and prepare the weight matrix

    if isinstance(W, np.ndarray) and W.shape != (m, m):
        raise ValueError(f"W must be a square matrix of size {m}x{m}")

    # Create the LP problem
    prob = pulp.LpProblem("Weighted_Minimal_Homologous_Cycle", pulp.LpMinimize)

    # Decision variables
    x_plus = [pulp.LpVariable(f"x_plus_{i}", lowBound=0) for i in range(m)]
    x_minus = [pulp.LpVariable(f"x_minus_{i}", lowBound=0) for i in range(m)]
    q = [pulp.LpVariable(f"q_{i}", cat="Continuous") for i in range(n)]

    # New variable b_i for each row sum
    b = [pulp.LpVariable(f"b_{i}", lowBound=0, cat="Continuous") for i in range(m)]

    # Objective function: minimize sum of b_i
    prob += pulp.lpSum(b)

    # Constraints
    # 1. Original homology constraints
    for i in range(m):
        prob += x_plus[i] - x_minus[i] == x_orig[i] + pulp.lpSum(
            boundary_matrix[i][j] * q[j] for j in range(n)
        )

    # 2. New constraints for row sums
    for i in range(m):
        # For each row i, sum_j W_ij(x_minus_j + x_plus_j) <= b_i
        prob += pulp.lpSum(W[i][j] * (x_minus[j] + x_plus[j]) for j in range(m)) <= b[i]

    # Solve the problem
    prob.solve()

    # Extract the solution
    x_solution = np.array(
        [pulp.value(x_plus[i]) - pulp.value(x_minus[i]) for i in range(m)]
    )

    return x_solution, prob.status, pulp.value(prob.objective)
