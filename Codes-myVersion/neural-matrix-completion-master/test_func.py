import numpy as np

def get_elements_vectorized(R, x_indices, y_indices ):
    ''' vectorized implementation of get_ratings functions. Gain 3x speedup '''
    n = x_indices.shape[0]
    m = y_indices.shape[0]
    values = np.zeros((n,m)).astype(np.float32)
    value_ind1, value_ind2 = np.meshgrid(x_indices, y_indices)
    ind1, ind2 = np.meshgrid(range(n), range(m))
    values[ind1.flatten(),ind2.flatten()] = R[value_ind1.flatten(), value_ind2.flatten()]
    return values


X = np.arange(15).reshape(5,3)

a = np.asarray([2,1,3])
b = np.asarray([1,2])

new_X = get_elements_vectorized(X, a, b)

print(X,"\n\n",new_X,"\n")