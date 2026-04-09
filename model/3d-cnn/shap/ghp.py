import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

def pairwise_ghp_and_ber(X, y, classes=None, verbose=True):
    """
    Pairwise MST estimation
    Returns:
      ber_estimate: float  -- aggregated BER estimate (see note below)
      pairwise: np.array  -- K x K matrix of pairwise GHP estimates (diagonal = 1)
      counts: np.array    -- number of points per class
      classes: np.array   -- array of class labels in index order

    Explanations/Notes:
    - For each pair of classes (i,j), an MST is built on the combined sample (classes i and j),
      and R_ij = number of MST edges connecting objects of different classes is counted.
    - The pairwise (normalized) statistic ghp_ij is calculated as:
          ghp_ij = 1 - R_ij / (N_ij - 1)
      where N_ij = n_i + n_j (number of points in the combined sample).
    - Aggregation into a single BER: here a simple mixture is used
          BER ≈ 1 - sum_{i<j} p_i * p_j * ghp_ij

    """
    X = np.asarray(X)
    y = np.asarray(y)
    assert X.shape[0] == y.shape[0]
    if classes is None:
        classes = np.unique(y)
    else:
        classes = np.asarray(classes)

    K = len(classes)

    counts = np.array([np.sum(y == c) for c in classes], dtype=int)
    total = len(y)
    priors = counts.astype(float) / total

    pairwise = np.eye(K, dtype=float)  
    for i in range(K):
        for j in range(i + 1, K):
            mask = (y == classes[i]) | (y == classes[j])
            Xi = X[mask]
            yi = y[mask]
            ni = Xi.shape[0]
            if ni <= 1:
                if verbose:
                    print(f"Пара ({classes[i]},{classes[j]}) имеет мало точек ({ni}) -> пропуск")
                pairwise[i, j] = pairwise[j, i] = 0.0
                continue

            D = cdist(Xi, Xi, metric='euclidean')
            np.fill_diagonal(D, np.inf)   

            mst = minimum_spanning_tree(csr_matrix(D)).tocoo()

            cross = 0
            for u, v in zip(mst.row, mst.col):
                if yi[u] != yi[v]:
                    cross += 1

            N_ij = ni
            denom = max(1, N_ij)
            D_tilde = 1.0 - 2*(cross / denom)
            low = 0.5-0.5*(D_tilde)**0.5
            high = 0.5-0.5*(D_tilde)

            pairwise[i, j] = pairwise[j, i] = (low+high)/2

            if verbose:
                print(f"Пара ({classes[i]},{classes[j]}): n={N_ij}, cross_edges={cross}, low={low:.4f}, high = {high:.4f}")

    comb = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            comb += priors[i] * priors[j] * pairwise[i, j]
    ber_estimate = 1.0 - comb

    if verbose:
        print(f"priors: {priors}")
        print(f"BER estimate (aggregated simple) = {ber_estimate:.6f}")

    return ber_estimate, pairwise, counts, classes

# data=[]
# for path in imgs:
#   img = np.load(path).reshape(-1, B)
#   idx = np.random.choice(np.arange(img.shape[0]),size=(1000,), replace = False)
#   rows= img[idx]
#   data.append(rows)
# data = np.concatenate(data,axis=0)