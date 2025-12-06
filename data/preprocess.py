# preprocess.py
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def generate_adjacency_matrix(group, output_file='adjacency_matrix.npy'):
    """
    Generate an adjacency matrix from a group of node arrays and save it in NumPy format.
    
    Parameters:
    group (np.ndarray): Array of arrays containing node groups, with possible extra nesting
    output_file (str): Path to save the adjacency matrix (default: 'adjacency_matrix.npy')
    
    Returns:
    np.ndarray: Generated adjacency matrix
    """
    # Extract the inner array to handle extra nesting
    group_inner = group[0] if group.ndim > 1 else group
    
    # Extract all unique nodes
    nodes = set()
    for arr in group_inner:
        # Flatten the array completely to get scalar values
        flattened = np.array(arr).flatten()
        nodes.update(flattened.tolist())  # Convert to list to ensure hashable elements
    nodes = sorted(list(nodes))  # Sort for consistent indexing
    n = len(nodes)  # Total number of nodes

    # Create adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=np.uint8)

    # Map nodes to indices (0-based indexing for the matrix)
    node_to_index = {node: idx for idx, node in enumerate(nodes)}

    # Fill adjacency matrix: connect nodes within the same group
    for arr in group_inner:
        group_nodes = np.array(arr).flatten()  # Flatten to get node list
        for i in range(len(group_nodes)):
            for j in range(i + 1, len(group_nodes)):
                node1, node2 = group_nodes[i], group_nodes[j]
                idx1, idx2 = node_to_index[node1], node_to_index[node2]
                adj_matrix[idx1, idx2] = 1
                adj_matrix[idx2, idx1] = 1  # Undirected graph

    # Save the adjacency matrix in NumPy format
    np.save(output_file, adj_matrix)

    return adj_matrix


def data_partition(Y_train, Y_test, A=None, 
                   X_aux_train=None,
                   unobserved_proportion=0.2, 
                   seed=None,
                   AUX=False):
    """
    Partitions nodes and time for training, validation, and test.
    If AUX=True, processes X_aux_train similarly to Y_train.
    A is optional and only used if provided.
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_nodes = Y_train.shape[0]
    num_unobserved = int(np.round(num_nodes * unobserved_proportion))
    num_observed = num_nodes - num_unobserved
    
    unobserved_indices = np.random.choice(num_nodes, size=num_unobserved, replace=False)
    Indices = np.ones(num_nodes, dtype=int)
    Indices[unobserved_indices] = 0
    observed_indices = np.where(Indices == 1)[0]
    
    # Process Y_train
    Y_train_observed_full = Y_train[observed_indices, :]
    time_horizon_train = Y_train_observed_full.shape[1]
    val_size = int(time_horizon_train * 0.2)
    train_size = time_horizon_train - val_size
    
    Y_train_observed = Y_train_observed_full[:, :train_size]
    Y_val_observed = Y_train_observed_full[:, train_size:]
    
    # [AUX] Process X_aux_train if provided and AUX=True
    if AUX:
        if X_aux_train is None:
            raise ValueError("X_aux_train must be provided when AUX=True")
        X_aux_observed_full = X_aux_train[observed_indices, :, :]
        X_aux_train_observed = X_aux_observed_full[:, :train_size, :]
        X_aux_val_observed = X_aux_observed_full[:, train_size:, :]
    else:
        X_aux_train_observed = None
        X_aux_val_observed = None
    
    # Process A and X_test
    A_observed = A[observed_indices, :][:, observed_indices] if A is not None else None
    M_test = np.ones_like(Y_test, dtype=int)
    M_test[unobserved_indices, :] = 0
    X_test = Y_test * M_test

    # Return values depend on AUX and A
    if AUX and A is not None:
        return (
            Y_train_observed, 
            Y_val_observed,
            X_aux_train_observed, 
            X_aux_val_observed,
            X_test, 
            A_observed, 
            M_test, 
            Indices
        )
    elif AUX:
        return (
            Y_train_observed, 
            Y_val_observed,
            X_aux_train_observed, 
            X_aux_val_observed,
            X_test, 
            M_test, 
            Indices
        )
    else:
        return (
            Y_train_observed, 
            Y_val_observed,
            X_test, 
            A_observed, 
            M_test, 
            Indices
        )

def train_construction(sample_size, Y_train_obs, A_obs=None, 
                       X_aux_train_obs=None,
                       h=24, N_sub=10, u=1,
                       AUX=False):
    """
    Constructs training samples. If AUX=True, includes x_aux_train.
    A_obs is optional and only used if provided.
    """
    num_nodes, time_horizon = Y_train_obs.shape

    if N_sub > num_nodes:
        raise ValueError("N_sub cannot be greater than the number of nodes in Y_train_obs")
    if h > time_horizon:
        raise ValueError("h cannot be greater than the time horizon in Y_train_obs")
    if u > N_sub:
        raise ValueError("u cannot be greater than N_sub")

    # Initialize base variables
    y_train = np.zeros((sample_size, N_sub, h))
    m_train = np.ones((sample_size, N_sub, h), dtype=int)
    a_train = np.zeros((sample_size, N_sub, N_sub)) if A_obs is not None else None
    x_train = np.zeros((sample_size, N_sub, h))
    
    # [AUX] Initialize auxiliary variables
    if AUX:
        if X_aux_train_obs is None:
            raise ValueError("X_aux_train_obs must be provided when AUX=True")
        F = X_aux_train_obs.shape[2]
        x_aux_train = np.zeros((sample_size, N_sub, h, F))
    else:
        x_aux_train = None

    for i in range(sample_size):
        node_indices = np.random.choice(num_nodes, size=N_sub, replace=False)
        start_t = np.random.randint(0, time_horizon - h + 1)
        time_slice = slice(start_t, start_t + h)

        # Main variable
        y_train[i] = Y_train_obs[node_indices, time_slice]
        if A_obs is not None:
            a_train[i] = A_obs[node_indices, :][:, node_indices]

        # Mask u nodes
        if u > 0:
            unobserved_local = np.random.choice(N_sub, size=u, replace=False)
            m_train[i, unobserved_local, :] = 0
        x_train[i] = y_train[i] * m_train[i]

        # [AUX] Auxiliary variables (sampled synchronously)
        if AUX:
            x_aux_train[i] = X_aux_train_obs[node_indices, time_slice, :]

    # Return values depend on AUX and A_obs
    if AUX and A_obs is not None:
        return y_train, x_aux_train, m_train, a_train, x_train
    elif AUX:
        return y_train, x_aux_train, m_train, x_train
    else:
        return y_train, m_train, a_train, x_train
    
def test_construction(sample_size, Y_test, X_test, A=None, Indices=None,
                      X_aux_test=None,
                      h=24, N_sub=10, u=1, seed=None,
                      AUX=False, return_indices=False):
    """
    Constructs test samples. If AUX=True, includes x_aux_test.
    A is optional and only used if provided.
    """
    if seed is not None:
        np.random.seed(seed)

    num_nodes, time_horizon = Y_test.shape

    if N_sub > num_nodes:
        raise ValueError("N_sub cannot be greater than num_nodes")
    if h > time_horizon:
        raise ValueError("h cannot be greater than time_horizon")
    if u > N_sub:
        raise ValueError("u cannot be greater than N_sub")

    observed_indices = np.where(Indices == 1)[0]
    unobserved_indices = np.where(Indices == 0)[0]

    if len(unobserved_indices) < u:
        raise ValueError(f"Not enough unobserved nodes: need {u}, have {len(unobserved_indices)}")
    if len(observed_indices) < (N_sub - u):
        raise ValueError(f"Not enough observed nodes: need {N_sub - u}, have {len(observed_indices)}")

    # Initialize base variables
    x_test = np.zeros((sample_size, N_sub, h))
    m_test = np.ones((sample_size, N_sub, h), dtype=int)
    a_test = np.zeros((sample_size, N_sub, N_sub)) if A is not None else None
    y_test = np.zeros((sample_size, N_sub, h))
    selected_nodes_all = np.zeros((sample_size, N_sub), dtype=int) if return_indices else None
    
    # [AUX] Initialize auxiliary variables
    if AUX:
        if X_aux_test is None:
            raise ValueError("X_aux_test must be provided when AUX=True")
        F = X_aux_test.shape[2]
        x_aux_test = np.zeros((sample_size, N_sub, h, F))
    else:
        x_aux_test = None

    for i in range(sample_size):
        selected_unobs = np.random.choice(unobserved_indices, size=u, replace=False)
        selected_obs = np.random.choice(observed_indices, size=N_sub - u, replace=False)
        selected_nodes = np.concatenate([selected_unobs, selected_obs])
        np.random.shuffle(selected_nodes)
        if return_indices:
            selected_nodes_all[i] = selected_nodes

        start_t = np.random.randint(0, time_horizon - h + 1)
        time_slice = slice(start_t, start_t + h)

        # Main variable
        y_test[i] = Y_test[selected_nodes, time_slice]
        x_test[i] = X_test[selected_nodes, time_slice]
        if A is not None:
            a_test[i] = A[selected_nodes, :][:, selected_nodes]

        # Build mask
        for j, node in enumerate(selected_nodes):
            if node in unobserved_indices:
                m_test[i, j, :] = 0
        x_test[i] = y_test[i] * m_test[i]

        # [AUX] Auxiliary variables (sampled synchronously)
        if AUX:
            x_aux_test[i] = X_aux_test[selected_nodes, time_slice, :]

    # Return values depend on AUX and A
    if AUX and A is not None:
        if return_indices:
            return x_test, x_aux_test, m_test, a_test, y_test, selected_nodes_all
        return x_test, x_aux_test, m_test, a_test, y_test
    elif AUX:
        if return_indices:
            return x_test, x_aux_test, m_test, y_test, selected_nodes_all
        return x_test, x_aux_test, m_test, y_test
    else:
        if return_indices:
            return x_test, m_test, a_test, y_test, selected_nodes_all
        return x_test, m_test, a_test, y_test
