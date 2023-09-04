import numpy as np
from sklearn.decomposition import PCA

def power_iteration(A, num_iterations: int):
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k

def main(data):
    m, n = data.shape
    cov = 1/m * np.dot(data.T, data)
    cov2 = np.cov(data, rowvar=False)
    eig_vec = power_iteration(cov2, 10)
    return eig_vec

def main2(data):
    pca = PCA()
    pca.fit(data)
    return pca.components_[0]
    
if __name__ == "__main__":
    data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], \
                    [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])
    print(main(data))
    print(main2(data))