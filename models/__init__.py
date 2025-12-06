# __init__.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq
from einops import repeat

def check_nan(tensor, name):
    """Check whether a tensor contains NaN values and report the result."""
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D-GCN.

    Args:
        adj_mx: Input adjacency matrix, PyTorch tensor of shape (n, n)

    Returns:
        random_walk_mx: Random walk matrix, PyTorch tensor of shape (n, n)
    """
    # Calculate degree matrix
    d = torch.sum(adj_mx, dim=1)
    
    # Calculate inverse degree matrix
    d_inv = torch.pow(d, -1)
    d_inv[torch.isinf(d_inv)] = 0.
    
    # Create diagonal matrix from d_inv
    d_mat_inv = torch.diag(d_inv)
    
    # Calculate random walk matrix: D^-1 * A
    random_walk_mx = torch.matmul(d_mat_inv, adj_mx)
    
    return random_walk_mx

def calculate_random_walk_matrix_3d(adj_mx):
    """
    Calculate the random walk adjacency matrix for a batch of graphs.
    Input and output are 3D PyTorch tensors with shape (total_sample_size, n, n), where n is the number of nodes.
    This is designed for D-GCN with batched input.

    Args:
        adj_mx: Input adjacency matrices, shape (total_sample_size, n, n), PyTorch tensor

    Returns:
        random_walk_mx: Random walk matrices, shape (total_sample_size, n, n), PyTorch tensor
    """
    # Validate input
    if adj_mx.dim() != 3 or adj_mx.size(1) != adj_mx.size(2):
        raise ValueError("Input adj_mx must be 3D with shape (total_sample_size, n, n)")

    total_sample_size, n, _ = adj_mx.size()

    # Initialize output tensor
    random_walk_mx = torch.zeros_like(adj_mx)

    # Process each sample using calculate_random_walk_matrix
    for sample_idx in range(total_sample_size):
        random_walk_mx[sample_idx] = calculate_random_walk_matrix(adj_mx[sample_idx])

    return random_walk_mx

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    with batch processing for adjacency matrices.
    """       
    def __init__(self, in_channels, out_channels, orders, activation='relu'): 
        """
        :param in_channels: Number of time steps.
        :param out_channels: Desired number of output features at each node in each time step.
        :param orders: The diffusion steps.
        :param activation: Activation function to use ('relu' or 'selu').
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)
        
    def _concat(self, x, x_):
        x_ = x_.unsqueeze(1)  # Add dimension for diffusion order
        return torch.cat([x, x_], dim=1)
        
    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :param A_q: The forward random walk matrix (batch_size, num_nodes, num_nodes)
        :param A_h: The backward random walk matrix (batch_size, num_nodes, num_nodes)
        :return: Output data of shape (batch_size, num_nodes, num_features)
        """
        batch_size = X.shape[0]  # batch_size
        num_node = X.shape[1]    # number of nodes
        input_size = X.size(2)   # time length
        supports = [A_q, A_h]    # List of batched adjacency matrices
        
        # Reshape X for batch matrix multiplication
        x0 = X  # (batch_size, num_nodes, input_size)
        x = torch.unsqueeze(x0, 1)  # (batch_size, 1, num_nodes, input_size)
        
        # Apply diffusion convolution for each batch and support matrix
        for support in supports:
            x1 = torch.bmm(support, x0)  # (batch_size, num_nodes, input_size)
            x = self._concat(x, x1)      # (batch_size, diffusion_steps, num_nodes, input_size)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.bmm(support, x1) - x0  # Chebyshev polynomial approximation
                x = self._concat(x, x2)
                x1, x0 = x2, x1
        
        # Reshape for final transformation
        x = x.permute(0, 2, 3, 1)  # (batch_size, num_nodes, input_size, num_matrices)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])
        
        # Apply learnable parameters and bias
        x = torch.bmm(x, self.Theta1.expand(batch_size, -1, -1))  # (batch_size, num_nodes, out_channels)
        x = x + self.bias  # Add bias
        
        # Apply activation
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)   
            
        return x
    
class IGNNK(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets with batch processing.
    x_s
     |GNN_3
    H_2 + H_1
     |GNN_2
    H_1
     |GNN_1
    x^y_m     
    """
    def __init__(self, input_dim, output_dim): 
        super(IGNNK, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dimension = 64
        self.order = 2

        self.GNN1 = D_GCN(self.input_dim, self.hidden_dimension, self.order)
        self.GNN2 = D_GCN(self.hidden_dimension, self.hidden_dimension, self.order)
        self.GNN3 = D_GCN(self.hidden_dimension, self.output_dim, self.order, activation='linear')

    def forward(self, X, A):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps)
        :param A: Adjacency matrix (batch_size, num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_nodes, num_timesteps)
        """  
        
        X_S = X  # (batch_size, num_nodes, num_timesteps)
        
        # Graph structure processing
        A_transposed = A.permute(0, 2, 1)  # (B, N, N)
        A_q = calculate_random_walk_matrix_3d(A_transposed)  # (B, N, N)
        A_h = calculate_random_walk_matrix_3d(A_transposed).permute(0, 2, 1)  # (B, N, N)
        
        # Pass through three D_GCN layers
        X_s1 = self.GNN1(X_S, A_q, A_h)  # (batch_size, num_nodes, hidden_dimension)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1  # Residual connection
        X_s3 = self.GNN3(X_s2, A_q, A_h)  # (batch_size, num_nodes, time_dimension)

        # Reshape back to original input format
        X_res = X_s3  # (batch_size, num_timesteps, num_nodes)
               
        return X_res

class NodeAdaptiveAugmentor(nn.Module):
    """
    Node-adaptive feature augmentation module.
    It converts auxiliary variables X_aux into a node-specific residual signal Δ and adds it to the main model output.
    """
    def __init__(self, F, D, N, use_gcn=False):
        """
        Args:
            F (int): Dimension of auxiliary variables (last dimension of X_aux).
            D (int): Hidden dimension (recommended 16–64).
            N (int): Number of nodes.
            use_gcn (bool): Whether to apply graph convolution on node preferences.
            gcn_hidden (int): GCN hidden dimension (if use_gcn=True).
        """
        super(NodeAdaptiveAugmentor, self).__init__()
        self.F = F
        self.D = D
        self.N = N
        self.use_gcn = use_gcn

        # Step 1: Encode auxiliary variables (X_aux -> H)
        self.W_e = nn.Linear(F, D)  # [F, D]

        # Step 3: Generate node weights (h_bar -> w_n)
        self.W_a = nn.Linear(D, D)  # [D, D]

        # Step 5: Project to scalar (H_enhanced -> Δ)
        self.W_p = nn.Linear(D, 1)  # [D, 1]

        # Optional: add graph convolution on node preferences
        if use_gcn:
            self.gcn = D_GCN(D, D, 2)  # Assuming D_GCN is already implemented


    def forward(self, X_aux, A=None):
        """
        Args:
            X_aux: (B, N, T, F)
            A: (B, N, N) adjacency matrix (required when use_gcn=True)

        Returns:
            delta: (B, N, T) residual signal
        """
        B, N, T, F = X_aux.shape
        assert N == self.N, f"Number of nodes does not match: {N} vs {self.N}"
        assert F == self.F, f"Auxiliary variable dimension does not match: {F} vs {self.F}"

        # ---------------- Step 1: encode auxiliary variables ----------------
        # (B, N, T, F) -> (B, N, T, D)
        H = torch.relu(self.W_e(X_aux))

        # ---------------- Step 2: compute node-level preferences ----------------
        # Time average: (B, N, T, D) -> (B, N, D)
        h_bar = H.mean(dim=2)  # dim=2 is the T dimension

        # ---------------- Optional: graph convolution to enhance node preferences ----------------
        if self.use_gcn and A is not None:
            # Assume D_GCN input (B, N, D) and output (B, N, D)
            A_transposed = A.permute(0, 2, 1)  # (B, N, N)
            A_q = calculate_random_walk_matrix_3d(A_transposed)  # (B, N, N)
            A_h = calculate_random_walk_matrix_3d(A_transposed).permute(0, 2, 1)  # (B, N, N)
            h_bar = self.gcn(h_bar, A_q, A_h)

        # ---------------- Step 3: generate node-adaptive weights w_n ----------------
        # (B, N, D) -> (B, N, D)
        w_n = torch.sigmoid(self.W_a(h_bar))  # [0,1] normalized importance

        # ---------------- Step 4: expand weights and apply enhancement ----------------
        # Expand w_n: (B, N, D) -> (B, N, 1, D)
        w_n_expanded = w_n.unsqueeze(2)  # insert along T dimension

        # Weighting: (B, N, T, D) * (B, N, 1, D) -> (B, N, T, D)
        H_enhanced = H * w_n_expanded

        # ---------------- Step 5: project to scalar residual signal ----------------
        # (B, N, T, D) -> (B, N, T, 1) -> (B, N, T)
        delta = self.W_p(H_enhanced).squeeze(-1)  # remove the last dimension

        # ---------------- Optional: add global bias ----------------
        delta = delta 

        return delta  # (B, N, T)
    

class Aux_DGCN(nn.Module):
    """
    Combined model integrating IGNNK and NodeAdaptiveAugmentor.
    The final output is the sum of the IGNNK reconstructed output and the NodeAdaptiveAugmentor's delta signal.
    """
    def __init__(self, input_dim, output_dim, aux_dim, num_nodes, hidden_dim=64, order=2, use_gcn=False):
        """
        Args:
            input_dim (int): Input dimension for IGNNK (main data).
            output_dim (int): Output dimension for IGNNK (reconstructed data).
            aux_dim (int): Dimension of auxiliary variables (F in NodeAdaptiveAugmentor).
            num_nodes (int): Number of nodes (N).
            hidden_dim (int): Hidden dimension for both models (D in NodeAdaptiveAugmentor).
            order (int): Order for D_GCN in IGNNK.
            use_gcn (bool): Whether to use GCN in NodeAdaptiveAugmentor.
        """
        super(Aux_DGCN, self).__init__()
        
        # Initialize IGNNK model
        self.ignnk = IGNNK(input_dim=input_dim, output_dim=output_dim)
        
        # Initialize NodeAdaptiveAugmentor model
        self.node_augmentor = NodeAdaptiveAugmentor(F=aux_dim, D=hidden_dim, N=num_nodes, use_gcn=use_gcn)
        
        # Store parameters
        self.num_nodes = num_nodes
        self.output_dim = output_dim

    def forward(self, X, A, X_aux):
        """
        Args:
            X (torch.Tensor): Input data for IGNNK, shape (batch_size, num_nodes, num_timesteps).
            A (torch.Tensor): Adjacency matrix, shape (batch_size, num_nodes, num_nodes).
            X_aux (torch.Tensor): Auxiliary input for NodeAdaptiveAugmentor, shape (batch_size, num_nodes, num_timesteps, aux_dim).
        
        Returns:
            torch.Tensor: Combined output, shape (batch_size, num_nodes, num_timesteps).
        """
        # Get IGNNK output
        X_res = self.ignnk(X, A)  # (batch_size, num_nodes, num_timesteps)
        
        # Get NodeAdaptiveAugmentor output
        delta = self.node_augmentor(X_aux, A)  # (batch_size, num_nodes, num_timesteps)
        
        # Sum the outputs
        final_output = X_res + delta  # (batch_size, num_nodes, num_timesteps)
        
        return final_output
    
class LSTMmodel(nn.Module):
    def __init__(self, num_nodes, num_timesteps, aux_dim, lstm_hidden=64, lstm_layers=2, dropout=0.0):
        super(LSTMmodel, self).__init__()
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.input_dim = num_nodes * (aux_dim + 1)  # N * (aux_dim + 1)

        # LSTM to process time series
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Map LSTM output to num_nodes dimension
        self.output_proj = nn.Linear(lstm_hidden, num_nodes)

    def forward(self, X, X_aux):
        # X: (B, N, T)
        # X_aux: (B, N, T, aux_dim)
        B, N, T = X.shape
        aux_dim = X_aux.shape[-1]

        # Step 1: expand X -> (B, N, T, 1)
        X_expanded = X.unsqueeze(-1)  # (B, N, T, 1)

        # Step 2: concatenate X and X_aux -> (B, N, T, aux_dim + 1)
        X_combined = torch.cat([X_expanded, X_aux], dim=-1)  # (B, N, T, aux_dim+1)

        # Step 3: transpose and reshape: merge N and (aux_dim+1) into the feature dimension
        # First to (B, T, N, aux_dim+1), then reshape to (B, T, N*(aux_dim+1))
        X_reshaped = X_combined.permute(0, 2, 1, 3).contiguous()  # (B, T, N, aux_dim+1)
        X_flattened = X_reshaped.view(B, T, -1)  # (B, T, N*(aux_dim+1))

        # Step 4: LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(X_flattened)  # lstm_out: (B, T, lstm_hidden)

        # Step 5: linear projection to num_nodes
        output = self.output_proj(lstm_out)  # (B, T, num_nodes)

        # Step 6: reshape to (B, num_nodes, T)
        output = output.permute(0, 2, 1)  # (B, num_nodes, T)

        return output
    
class DGCN_LSTM(nn.Module):
    """
    Combined model: LSTM followed by DGCN (IGNNK).
    LSTM processes X and X_aux, its output becomes input to IGNNK with A.
    """
    def __init__(self, input_dim, output_dim, aux_dim, num_nodes, lstm_hidden=64, lstm_layers=2, dropout=0.0):
        """
        Args:
            input_dim (int): Input dimension for LSTM (num_timesteps).
            output_dim (int): Output dimension for IGNNK (num_timesteps).
            aux_dim (int): Dimension of auxiliary variables.
            num_nodes (int): Number of nodes (N).
            lstm_hidden (int): Hidden dimension for LSTM.
            lstm_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate for LSTM.
        """
        super(DGCN_LSTM, self).__init__()
        
        # LSTM part
        self.lstm = LSTMmodel(num_nodes=num_nodes, num_timesteps=input_dim, aux_dim=aux_dim, 
                              lstm_hidden=lstm_hidden, lstm_layers=lstm_layers, dropout=dropout)
        
        # DGCN (IGNNK) part
        self.dgcn = IGNNK(input_dim=output_dim, output_dim=output_dim)  # input_dim=output_dim since LSTM outputs (B,N,T)

    def forward(self, X, A, X_aux):
        """
        Args:
            X (torch.Tensor): Input data, shape (batch_size, num_nodes, num_timesteps).
            A (torch.Tensor): Adjacency matrix, shape (batch_size, num_nodes, num_nodes).
            X_aux (torch.Tensor): Auxiliary input, shape (batch_size, num_nodes, num_timesteps, aux_dim).
        
        Returns:
            torch.Tensor: Output, shape (batch_size, num_nodes, num_timesteps).
        """
        # Feed to DGCN
        dgcn_out = self.dgcn(X, A)  # (batch_size, num_nodes, num_timesteps)
        
        # LSTM output
        lstm_out = self.lstm(dgcn_out, X_aux)  # (batch_size, num_nodes, num_timesteps)
        
        return lstm_out
    
import torch

def KNN(X, A, m, k=3, eps=1e-8, device=None):
    """
    Impute fully missing nodes (where m is all 0) in time-series node data X.
    Assumes nodes are either fully missing (m == 0) or fully valid (m == 1).

    Global-mean mode (k == 0): compute means over time steps and fill missing values
    with the corresponding time-step mean.
    KNN mode (k > 0): use adjacency matrix A to find up to k valid neighbors and
    fill with their mean.

    Optimized by vectorizing over batches and time steps to reduce loops and
    improve performance. In KNN mode, prints the proportion of nodes that fall
    back to global mean.

    Args:
        X (torch.Tensor): Input tensor of shape (B, N, T), where B is batch size,
                          N is number of nodes, and T is number of time steps.
                          Missing values are represented by torch.nan.
        A (torch.Tensor): Adjacency matrix of shape (B, N, N) or (N, N), with
                          entries 0 or 1 indicating connections.
        m (torch.Tensor): Mask tensor of shape (B, N, T); 0 indicates positions
                          to be imputed, 1 indicates valid values.
        k (int or None): In KNN mode, number of neighbors with m == 1 to use.
                         If k == 0, use global-mean mode instead.
        eps (float): Small epsilon to avoid division by zero.
        device (str or None): Compute device, default None (CPU); can be 'cuda'
                              to use GPU.

    Returns:
        torch.Tensor: Imputed tensor of shape (B, N, T).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = X.to(device)
    A = A.to(device)
    m = m.to(device)

    # Ensure A is 3D with shape (B, N, N)
    if A.dim() == 2:
        A = A.unsqueeze(0).expand(X.size(0), -1, -1)

    B, N, T = X.shape

    # Check that m has a matching shape
    if m.shape != (B, N, T):
        raise ValueError(f"m shape {m.shape} does not match X shape {X.shape}")

    # Check node states: only fully missing or fully valid nodes are allowed
    fully_missing_nodes = torch.all(m == 0, dim=2)  # (B, N), True indicates a fully missing node
    fully_valid_nodes = torch.all(m == 1, dim=2)   # (B, N), True indicates a fully valid node
    if not torch.all(fully_missing_nodes | fully_valid_nodes):
        raise ValueError("All nodes must be either fully missing (m == 0) or fully valid (m == 1)")

    # If there are no positions to impute, return directly
    if not fully_missing_nodes.any():
        print("No missing values to impute")
        return X.clone()

    # Create output tensor
    X_filled = X.clone()

    # Compute global mean (along time steps) for global-mean mode and as a fallback for KNN mode
    valid_counts = m.sum(dim=1).clamp(min=eps)  # (B, T)
    global_sums = (X * m).sum(dim=1)  # (B, T)
    global_means = global_sums / valid_counts  # (B, T)
    global_means = torch.where(torch.isnan(global_means), torch.zeros_like(global_means), global_means)

    # --- Global-mean mode ---
    if k == 0:
        print("Global mean mode")
        # Get indices of fully missing nodes
        batch_idx, node_idx = torch.where(fully_missing_nodes)  # (num_missing,), (num_missing,)
        if batch_idx.numel() > 0:
            # Fill each missing node with the corresponding batch-wise global mean
            X_filled[batch_idx, node_idx, :] = global_means[batch_idx, :]  # (num_missing, T)

    # --- KNN mode ---
    else:
        print("KNN mode")
        # Precompute neighbor index tensor
        neighbor_mask = A.bool()  # (B, N, N)

        # Get indices of all fully missing nodes
        batch_idx, node_idx = torch.where(fully_missing_nodes)  # (num_missing,), (num_missing,)

        total_missing = batch_idx.numel()
        if total_missing > 0:
            # Get neighbor mask for all fully missing nodes
            neighbor_indices = neighbor_mask[batch_idx, node_idx]  # (num_missing, N)
            valid_neighbor_mask = fully_valid_nodes[batch_idx] & neighbor_indices  # (num_missing, N)

            # Compute the proportion of nodes that fall back to global mean
            has_valid_neighbors = valid_neighbor_mask.any(dim=1)  # (num_missing,)
            num_degraded = total_missing - has_valid_neighbors.sum().item()
            degradation_proportion = num_degraded / total_missing if total_missing > 0 else 0.0
            print(f"Proportion of nodes degraded to global mean: {degradation_proportion:.4f}")

            # Initialize fill values
            fill_values = torch.zeros(total_missing, T, device=device)  # (num_missing, T)

            # Handle nodes with valid neighbors
            if has_valid_neighbors.any():
                valid_batch_idx = batch_idx[has_valid_neighbors]
                valid_node_idx = node_idx[has_valid_neighbors]
                valid_neighbor_mask_subset = valid_neighbor_mask[has_valid_neighbors]  # (num_valid, N)

                # Get up to k valid neighbors
                _, topk_indices = torch.topk(valid_neighbor_mask_subset.float(), k=k, dim=1, largest=True)  # (num_valid, k)
                # Convert to global node indices
                batch_indices = valid_batch_idx[:, None].expand(-1, k)  # (num_valid, k)
                neighbor_idx = topk_indices  # (num_valid, k)

                # Get neighbor data
                neighbor_data = X[batch_indices, neighbor_idx]  # (num_valid, k, T)
                neighbor_m = m[batch_indices, neighbor_idx]  # (num_valid, k, T)

                # Compute mean
                valid_neighbor_data = neighbor_data * neighbor_m  # (num_valid, k, T)
                neighbor_sum = valid_neighbor_data.sum(dim=1)  # (num_valid, T)
                neighbor_count = neighbor_m.sum(dim=1).clamp(min=eps)  # (num_valid, T)
                fill_values[has_valid_neighbors] = neighbor_sum / neighbor_count  # (num_valid, T)

            # Handle nodes without valid neighbors (fall back to global mean)
            no_valid_neighbors = ~has_valid_neighbors
            if no_valid_neighbors.any():
                degraded_batch_idx = batch_idx[no_valid_neighbors]
                fill_values[no_valid_neighbors] = global_means[degraded_batch_idx]

            # Fill missing nodes
            X_filled[batch_idx, node_idx, :] = fill_values

    return X_filled

# Model registry
MODEL_REGISTRY = {"DGCN": IGNNK, "AUX_DGCN": Aux_DGCN, "LSTM": LSTMmodel, "DGCN_LSTM": DGCN_LSTM, "KNN": KNN}
