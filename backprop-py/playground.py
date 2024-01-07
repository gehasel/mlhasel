# %%
import numpy as np
a = np.array([1, 2, 3])
b = a.reshape(-1, 1)
# %%
a
# %%
b
# %%
b.reshape(-1, 1)
# %%
np.matmul(np.array([[1, 2, 3]]), b)
# %%
def test(a):
    a = a.reshape(-1, 1)
    print(a.shape)

a.shape
# %%
test(a)
# %%
a.shape
# %%
import numpy as np

# Example 2D matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([[11], [22], [33]])
print(matrix)
print(vector)
# Reshape the matrix to stack all columns into a single column
reshaped_matrix = matrix.T.flatten()
print(reshaped_matrix)
reshaped_vector = vector.T.flatten()
print(reshaped_vector)
print(flat_combined_vector := np.concatenate((reshaped_matrix, reshaped_vector)).reshape(-1, 1))
# %%
flat_combined_vector.flatten()
# %%
num_matrix_elems = np.prod(matrix.shape)
flattened_matrix = flat_combined_vector[:num_matrix_elems]
flattened_vector = flat_combined_vector[num_matrix_elems:]
restored_matrix = flattened_matrix.reshape(matrix.shape).T
restored_vector = flattened_vector.reshape(vector.shape)

print(restored_matrix)
print(restored_vector)
# %%
import numpy as np

a = np.array(
    [[1, 2],
     [3, 4]]
     )
b = np.array([[11], [22]])
np.fill_diagonal(a, b.flatten() + a.diagonal())
a
# %%
import numpy as np

a = np.array(
    [[1, 2],
     [3, 4]]
     )
b = np.array([[11], [22]])

a @ b
# %%
import numpy as np

def create_matrix(a, k):
    n = len(a)
    # Create the upper part of the matrix
    upper_matrix = np.zeros((n * k, k))
    for i in range(k):
        upper_matrix[i*n:(i+1)*n, i] = a[:, 0]  # Fill the ith column with the vector a

    # Create the lower k x k identity matrix
    lower_matrix = np.eye(k)

    # Stack the two matrices vertically
    result_matrix = np.vstack((upper_matrix, lower_matrix))

    return result_matrix

# Example usage
a = np.array([[1], [2]])  # Column vector a
k = 3  # Parameter k
result = create_matrix(a, k)
print(result)

# %%
f1
functions_after = [f3, f4]


# %%
id(4)
# %%
(3,4) == (..., 4)
# %%
import numpy as np

# Example list of column vectors
column_vectors = [np.array([[1], [2], [33]]), np.array([[3], [4]]), np.array([[5], [6]])]

# Stack them vertically to get a new column vector
stacked_vector = np.vstack(column_vectors)

print(stacked_vector)

# %%
import numpy as np

def get_weight_vector(weight_matrix, bias_vector):
    """
    Return a column vector of shape (number of weights, 1)
    """
    return np.concatenate((weight_matrix.T.flatten(), bias_vector.T.flatten())).reshape(-1, 1)


def set_weight_vector(weight_matrix, bias_vector, stacked_weight_vector: np.ndarray):
    """
    stacked_weight_vector: column vector of shape (number of weights, 1)
    """
    flat_weight_vector = stacked_weight_vector.flatten()
    # print(f"{flat_weight_vector}")
    num_matrix_elems = np.prod(weight_matrix.shape)
    # print(f"{num_matrix_elems=}")

    flattened_weight_matrix = flat_weight_vector[:num_matrix_elems]
    # print(f"{flattened_weight_matrix=}")

    flattened_bias_vector = flat_weight_vector[num_matrix_elems:]
    # print(f"{flattened_bias_vector=}")

    weight_matrix = flattened_weight_matrix.reshape(weight_matrix.shape)
    # print(f"{weight_matrix=}")

    bias_vector = flattened_bias_vector.reshape(bias_vector.shape)

    return (weight_matrix, bias_vector)
    # print(f"{bias_vector=}")
# %%
W = np.arange(15).reshape(5, 3)
b = np.array([1515, 1616, 1717])

new_weight_vector = - np.arange(18)
# %%
print(W)
print(b)
print(get_weight_vector(W, b))
print(new_weight_vector)
# %%
W, b = set_weight_vector(W, b, new_weight_vector)
# %%
class C1:
    def __init__(self, elem: C2):
        self.C2 = C2

class C2:
    def __init__(self, elem: C1):
        self.C1 = C1

# %%
import numpy as np

def one_hot_encode(arr):
    # Create an array of zeros with the required shape (n, 10)
    one_hot = np.zeros((arr.size, 10))
    
    # Use numpy's advanced indexing to set the appropriate indices to 1
    one_hot[np.arange(arr.size), arr] = 1
    
    return one_hot

# Example usage
arr = np.array(list(reversed([0, 1, 2, 3, 4])))
encoded_arr = one_hot_encode(arr)
print(encoded_arr)


# %%
not 99
# %%
import numpy as np

class MatrixHandler:
    def __init__(self, matrix):
        self.matrix = matrix

    def get_flattened(self):
        # Flatten the matrix with rows from the transposed matrix stacked
        return self.matrix.reshape(-1)

    def set_matrix(self, flattened):
                # Assuming the flattened array has the correct size (n*m)
        n, m = self.matrix.shape
        # Reshape the flattened array back to (n, m)
        self.matrix = flattened.reshape(n, m)

# Example usage
matrix = np.array([[1, 2, 3], [4, 5, 6]])
handler = MatrixHandler(matrix)

# Get flattened matrix
flattened = handler.get_flattened()
print(flattened)

# Set matrix using a new flattened array
new_flattened = np.array([1, 0, 0, 9, 9, 9])
handler.set_matrix(new_flattened)

# Output the new matrix
print(handler.matrix)
# %%
import numpy as np


class MyHandler:
    def __init__(self, matrix, bias_vector):
        self.weight_matrix = matrix
        self.bias_vector = bias_vector

    @property
    def weight_vector(self):
        """
        Return a column vector of shape (number of weights, 1)
        """
        return np.concatenate((self.weight_matrix.T.flatten(), self.bias_vector.T.flatten())).reshape(-1, 1)

    @weight_vector.setter
    def weight_vector(self, flattened: np.ndarray):
        """
        stacked_weight_vector: column vector of shape (number of weights, 1)
        """
        matrix_portion = flattened[0:self.weight_matrix.size]
        bias_portion = flattened[self.weight_matrix.size:]

        n, m = self.weight_matrix.shape
        # note how m and n are switched here:
        self.weight_matrix = matrix_portion.reshape(m, n).T
        self.bias_vector = bias_portion.reshape(-1, 1)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias_vector = np.array([[22], [33]])
handler = MyHandler(matrix, bias_vector)
flattened = handler.weight_vector
print(flattened)

# Set matrix using a new flattened array
new_flattened = np.array([1, -1, 2, -2, 3, -3, 22, 33])
handler.weight_vector = new_flattened
print(handler.weight_matrix)
print(handler.bias_vector)
# %%
import numpy as np
a = np.array([[1], [2], [-1]])
dim = a.shape[0]
J = np.zeros((dim, dim))
np.fill_diagonal(J, a > 0)
r = J.T
r
# %%
