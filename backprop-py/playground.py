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
