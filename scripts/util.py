import torch
import numpy as np
# from factor_analyzer.rotator import Rotator
from scipy.sparse import diags
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

def display_image(testset, img_idx, ax, title='', wnid_to_class=None):
    image, label = testset[img_idx][:2]
    # Set the title only if wnid_to_class is not None
    if wnid_to_class is not None:
        label = wnid_to_class[testset.classes[label]]
        title = label + title
    image = image.numpy()  # Convert to numpy array
    ax.imshow(np.transpose(image, (1, 2, 0)))  # Reorder dimensions for display
    if title:
        ax.set_title(title, fontsize=20)

def get_top_n_elements_per_dim(csr, n=5):
    top_indices = []
    top_values = []

    for i in range(csr.shape[0]):
        # Convert the sparse row/column to a dense array
        dense_line = csr[i].toarray()[0]

        # Use argpartition to find the indices of the top n elements
        top_n_idx = np.argpartition(-dense_line, n)[:n]

        # Optional: sort the indices to get the top n elements in order
        top_n_idx_sorted = top_n_idx[np.argsort(-dense_line[top_n_idx])]

        # Get the top n values using the sorted indices
        top_n_values = dense_line[top_n_idx_sorted]

        top_indices.append(top_n_idx_sorted)
        top_values.append(top_n_values)

    return top_indices, top_values

def get_top_n_elements(csr, n=5, axis=1):
    if axis == 1:  # For rows
        return get_top_n_elements_per_dim(csr, n)
    elif axis == 0:  # For columns
        # For columns, transpose the matrix to use the same function
        csr_transposed = csr.transpose()
        return get_top_n_elements_per_dim(csr_transposed, n)
    else:
        raise ValueError("Axis must be 0 (columns) or 1 (rows)")

def glaplacian(A, eta=1):
    # Assuming A is a numpy array or a scipy sparse matrix
    # A = csr_matrix(A)  # Ensure A is in sparse format for efficiency
    A_abs = abs(A)
    deg_row = np.array(A_abs.sum(axis=1)).flatten()  # Row sums
    deg_col = np.array(A_abs.sum(axis=0)).flatten()  # Column sums
    
    if eta > 0:
        tau_row = eta * np.mean(deg_row)
        tau_col = eta * np.mean(deg_col)
    else:
        if np.any(np.concatenate((deg_row, deg_col)) == 0):
            raise ValueError("Cannot use Laplacian because some nodes are isolated. Set either \"regularize=True\" or \"laplacian=False\" option.")
        tau_row = tau_col = 0
    
    # Diagonal matrices for normalization
    D_row = diags(1 / np.sqrt(deg_row + tau_row))
    D_col = diags(1 / np.sqrt(deg_col + tau_col))
    
    # Compute the Laplacian
    L = D_row @ A @ D_col
    
    return L

# Function to create row and column names similar to R's paste functionality
def create_names(prefix, n):
    return [f"{prefix}{i}" for i in range(1, n+1)]

def q_matrix(distances):
    m, n = distances.shape
    row_dists = distances.sum(axis=1)
    col_dists = distances.sum(axis=0)
    q_matrix = distances - np.add.outer(row_dists / m, col_dists / n)
    return q_matrix

def to_lower_triangle(matrix):
    lower_triangle = [[0]]
    for i in range(1, len(matrix)):
        row = []
        for j in range(i):
            row.append(matrix[i][j])
        row.append(0)
        lower_triangle.append(row)
    return lower_triangle


def a_nj(D, max_itr=100):
    import pandas as pd
    D = pd.DataFrame(D)
    # Set row and column names
    row_names = ['r' + str(i) for i in range(1, D.shape[0]+1)]
    col_names = ['b' + str(i) for i in range(1, D.shape[1]+1)]
    D.index = row_names
    D.columns = col_names
    # Initialize twigs list
    twigs = []
    itr = 0
    while (D.shape[0]+D.shape[1])>=3 and (itr < max_itr) and (set(row_names)!=set(col_names)):
        itr += 1
        Q = q_matrix(D.values)
        mask = np.equal.outer(D.index, D.columns)
        Q[mask] = np.inf
        min_val = np.unravel_index(np.argmin(Q, axis=None), Q.shape)
        i, j = min_val
        merged_nodes = [D.index[i], D.columns[j]]
        merged_rows = [i for i in D.index if i in merged_nodes]
        merged_cols = [j for j in D.columns if j in merged_nodes]
        new_node = "P" + str(len(twigs)+1)
        d_sibling = D.iat[i, j]
        if d_sibling < 0 and abs(d_sibling)<=1:
            d_sibling = 0.2
        twig = "({}:{},{}:{}){}".format(D.index[i], round(1/2*d_sibling,3), D.columns[j], round(1/2*d_sibling,3), new_node)
        twigs.append(twig)
        # Update D by removing merged rows and columns and adding a new row/column for the merged node
        remained_indices = [idx for idx in D.index if idx not in merged_nodes]
        remained_columns = [col for col in D.columns if col not in merged_nodes]
        remained = D.loc[remained_indices, remained_columns]
        new_row = pd.Series(np.mean(D.loc[merged_rows, remained_columns].values, axis=0) - d_sibling/2, index=remained_columns)
        new_col = pd.Series(np.mean(D.loc[remained_indices, merged_cols].values, axis=1) - d_sibling/2, index=remained_indices)
        new_col[new_node] = 0
        if len(remained.columns)== 0 or len(remained.index)==0:
            remained = pd.DataFrame(index= remained_indices+[new_node], columns=remained_columns+[new_node])
        remained.loc[new_node] = new_row
        remained[new_node] = new_col
        D = remained
    return D, twigs



def varimax(Phi, gamma = 1.0, q = 25, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, linalg
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = linalg.svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, np.diag(np.diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: 
            break
    return dot(Phi, R)

def varimax_with_rotation(Phi, gamma = 1.0, q = 25, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, linalg
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = linalg.svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, np.diag(np.diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: 
            break
    return dot(Phi, R), R


# embeddings = abs(embeddings)
def glaplacian_abs(A, eta=2):
    A_abs = abs(A)
    deg_row = A_abs.sum(dim=1, dtype=torch.float32) 
    deg_col = A_abs.sum(dim=0, dtype=torch.float32)
    
    if eta > 0:
        tau_row = eta * torch.mean(deg_row)
        tau_col = eta * torch.mean(deg_col)
    else:
        if np.any(np.concatenate((deg_row, deg_col)) == 0):
            raise ValueError("Cannot use Laplacian because some nodes are isolated. Set either \"regularize=True\" or \"laplacian=False\" option.")
        tau_row = tau_col = 0
    
    # Diagonal matrices for normalization
    D_row = diags((1 / np.sqrt(deg_row + tau_row)).numpy())
    D_col = diags((1 / np.sqrt(deg_col + tau_col)).numpy())
    L = D_row @ A @ D_col
    return L


def find_item(items, scores):    
    cumulative_scores = {}
    # Aggregate scores by items
    for item, score in zip(items, scores):
        if item in cumulative_scores:
            cumulative_scores[item] += score
        else:
            cumulative_scores[item] = score
    # Determine the item with the highest total score
    max_item = max(cumulative_scores, key=cumulative_scores.get)
    return max_item

def varimax_and_sign_flip(L, k=50):
    U, S, Vt = randomized_svd(L, n_components=k)
    U_rotated = varimax(U)
    Vt_rotated = varimax(Vt.T)
    sign_Z = np.diag(np.where(np.mean(U_rotated**3, axis=0) >= 0, 1, -1))
    Z_hat = np.dot(U_rotated, sign_Z)
    sign_Y = np.diag(np.where(np.mean(Vt_rotated**3, axis=0) >= 0, 1, -1))
    Y_hat = np.dot(Vt_rotated, sign_Y)
    B = Z_hat.T.dot(L).dot(Y_hat)
    return Z_hat, Y_hat, B

def plot_top_images_per_cluster(Z_hat, raw_dataset, wnid_to_class, file_path, n=9, nrows=3, ncols=6):
    # Get the top n positive and negative images for each cluster
    top_ims, im_scores = get_top_n_elements(csr_matrix(Z_hat), n=n, axis=0)
    top_neg_ims, neg_im_scores = get_top_n_elements(csr_matrix(-Z_hat), n=n, axis=0)
    
    # Total number of clusters
    nclusters = Z_hat.shape[1]
    
    for cluster_idx in range(nclusters):
        plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
        for img_sub_idx in range(n):  # Iterate over the top n images for each cluster
            row = img_sub_idx // 3
            col = img_sub_idx % 3
            
            # Positive images
            ax_pos = plt.subplot(nrows, ncols, row * ncols + col + 1)
            show_idx_pos = top_ims[cluster_idx][img_sub_idx]  # Get the index for the current positive image
            display_image(raw_dataset, show_idx_pos, ax_pos,'', wnid_to_class)
            ax_pos.set_xticks([])  # Remove x-axis ticks
            ax_pos.set_yticks([])  # Remove y-axis ticks

            # Negative images
            ax_neg = plt.subplot(nrows, ncols, row * ncols + col + 4)  # Adjust the subplot index for negative images
            show_idx_neg = top_neg_ims[cluster_idx][img_sub_idx]  # Get the index for the current negative image
            display_image(raw_dataset, show_idx_neg, ax_neg,'',wnid_to_class)
            ax_neg.set_xticks([])  # Remove x-axis ticks
            ax_neg.set_yticks([])  # Remove y-axis ticks

        plt.tight_layout()
        plt.suptitle(f'Cluster {cluster_idx + 1}', fontsize=20)
        plt.subplots_adjust(top=0.9)  # Adjust the top spacing to accommodate the suptitle

        # Save the figure
        img_path = f'{file_path}cluster_{cluster_idx + 1}.png'
        plt.savefig(img_path)
        plt.close()  # Close the figure to free memory

def plot_top_pos_images_per_cluster(Z_hat, raw_dataset, wnid_to_class, file_path, n=9, nrows=3, ncols=3):
    # Get the top n positive images for each cluster
    top_ims, _ = get_top_n_elements(csr_matrix(Z_hat), n=n, axis=0)
    
    # Total number of clusters
    nclusters = Z_hat.shape[1]
    
    for cluster_idx in range(nclusters):
        plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
        for img_sub_idx in range(n):  # Iterate over the top n images for each cluster
            row = img_sub_idx // 3
            col = img_sub_idx % 3
            
            # Positive images
            ax_pos = plt.subplot(nrows, ncols, row * ncols + col + 1)
            show_idx_pos = top_ims[cluster_idx][img_sub_idx]  # Get the index for the current positive image
            display_image(raw_dataset, show_idx_pos, ax_pos, '', wnid_to_class)
            ax_pos.set_xticks([])  # Remove x-axis ticks
            ax_pos.set_yticks([])  # Remove y-axis ticks

        plt.tight_layout()
        
        # Save the figure
        img_path = f'{file_path}cluster_{cluster_idx + 1}.png'
        plt.savefig(img_path)
        plt.close()  # Close the figure to free memory

def center_matrix(A):
    n, d = A.shape
    # Compute means
    mu_r = np.mean(A, axis=1, keepdims=True)  # Row means (n x 1)
    mu_c = np.mean(A, axis=0, keepdims=True)  # Column means (1 x d)
    mu = np.mean(A)                           # Grand mean (scalar)
    # Compute centered matrix
    A_hat = A - mu_r @ np.ones((1, d)) - np.ones((n, 1)) @ mu_c + mu
    return A_hat

def show_images_grid(indices, dataset):
    """
    Function to display a 3x3 grid of images from the dataset based on provided indices.
    
    Args:
    indices (list): List of indices for the images to display.
    dataset (Dataset): The dataset object to retrieve images from.
    """
    # Create a 3x3 plot
    _, axs = plt.subplots(3, 3, figsize=(7, 7))
    
    # Loop through the grid and display each image
    for ax, idx in zip(axs.flatten(), indices):
        img = dataset[idx][0]  # Assuming the image is at index 0 for each dataset item
        ax.imshow(img)
        ax.axis('off')  # Hide axis labels

    plt.tight_layout()  # Adjust the layout to avoid overlapping
    plt.show()



    

