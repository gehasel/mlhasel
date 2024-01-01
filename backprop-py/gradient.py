import numpy as np
from typing import List, Optional
from functools import reduce
from IPython.display import display


class Layer:
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return super().__str__()

    def get_weight_vector(self) -> np.ndarray:
        raise NotImplementedError

    def set_weight_vector(self, stacked_weight_vector: np.ndarray) -> None:
        raise NotImplementedError

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """
        forward pass

        a:  column vector of shape (input_dim, 1) which is the output of the previous layer,
            if this is the first layer, a == x
        """
        raise NotImplementedError

    def params_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: output of the previous layer in forward pass
        """
        raise NotImplementedError

    def input_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: output of the previous layer in forward pass
        """
        raise NotImplementedError


class Lin(Layer):
    def __init__(self, weight_matrix: np.ndarray, bias_vector: np.ndarray, name: Optional[str] = None):
        """
        weight_matrix of shape (number of inputs, number of outputs)
        bias_vector of shape (number of outputs, 1)
        """
        super().__init__()
        self.weight_matrix = weight_matrix
        self.bias_vector = bias_vector
        self.input_dim = self.weight_matrix.shape[0]
        self.output_dim = self.weight_matrix.shape[1]
        self.num_params = self.input_dim * self.output_dim + self.output_dim

    def get_weight_vector(self):
        """
        Return a column vector of shape (number of weights, 1)
        """
        return np.concatenate((self.weight_matrix.T.flatten(), self.bias_vector.T.flatten())).reshape(-1, 1)

    def set_weight_vector(self, stacked_weight_vector):
        """
        stacked_weight_vector: column vector of shape (number of weights, 1)
        """
        flat_weight_vector = stacked_weight_vector.flatten()
        num_matrix_elems = np.prod(self.weight_matrix.shape)
        flattened_weight_matrix = flat_weight_vector[:num_matrix_elems]
        flattened_bias_vector = flat_weight_vector[num_matrix_elems:]
        self.weight_matrix = flattened_weight_matrix.T.reshape(self.weight_matrix.shape)
        self.bias_vector = flattened_bias_vector.reshape(self.bias_vector.shape)

    def __call__(self, a: np.ndarray):
        """
        a is a column vector, if this is the first layer, a == x
        """
        assert a.shape == (self.input_dim, 1)
        r = (self.weight_matrix.T @ a) + self.bias_vector
        assert r.shape == (self.output_dim, 1)
        return r

    def params_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: input of the forward pass at this layer
        """
        assert a.shape == (self.input_dim, 1)
        upper_matrix = np.zeros((self.input_dim * self.output_dim, self.output_dim))
        for i in range(self.output_dim):
            # Fill the ith column with the vector a:
            upper_matrix[i * self.input_dim:(i + 1) * self.input_dim, i] = a[:, 0]
        lower_matrix = np.eye(self.output_dim)  # grad of the bias terms
        result_matrix = np.vstack((upper_matrix, lower_matrix))
        assert result_matrix.shape == (self.num_params, self.output_dim)
        return result_matrix

    def input_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: input of the forward pass at this layer
        """
        assert a.shape == (self.input_dim, 1)
        result_matrix = self.weight_matrix
        assert result_matrix.shape == (self.input_dim, self.output_dim)
        return result_matrix


class Softmax(Layer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def __call__(self, z: np.ndarray):
        z_max = np.max(z, axis=0)
        z_norm = z - z_max
        r = np.divide(np.exp(z_norm), np.sum(np.exp(z_norm), axis=0))
        return r

    def input_gradient(self, z_l: np.ndarray):
        """
        The transpose of the jacobian where every derivative in the jacobian
        evaluated z_l
        """
        exp_z = np.exp(z_l)
        normalized_exp_z = exp_z / np.sum(exp_z)
        J = -np.outer(normalized_exp_z, normalized_exp_z)
        # for some reason, we must flatten normalized... here,
        # otherwise it only adds the first element of b to the diagonal, not elementwise
        np.fill_diagonal(J, normalized_exp_z.flatten() + J.diagonal())
        # J is now the jacobian where every partial derivative has already evaluated z_l
        # Although symmetric, I transpose for consistency
        r = J.T
        return r

    def get_weight_vector(self):
        return np.array([])

    def set_weight_vector(self, flat_vector):
        pass


class CE(Layer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def __call__(self, y: np.ndarray, y_pred: np.ndarray):
        """
        y_pred == f_z_L == last_activation_function(z_L)
        """
        return -(y.T @ np.log(y_pred))

    def input_gradient(self, y: np.ndarray, y_pred: np.ndarray):
        """
        transpose of the jacobian with the derivatives evaluated with f_z_L.
        Multiply last, since matrix multiplication order is reversed with
        gradients vs. jacobians.
        """
        # make sure f_z_L and y are a row vectors (does not change it outside this function):
        f_z_L = y_pred.reshape(1, -1)
        y = y.reshape(1, -1)
        J = -np.divide(y, f_z_L)
        r = J.T
        return r

    def get_weight_vector(self):
        return np.array([])

    def set_weight_vector(self, flat_vector):
        pass


def compose(f, g):
    return lambda x: f(g(x))


class NN:
    def __init__(self, functions: List[callable], loss: callable):
        self.functions = functions
        self.composite = reduce(
            compose, reversed(functions)
        )  # like foldl, but order does not matter, since composition is associative
        self.loss = loss

    def __call__(self, x: np.ndarray):
        return self.composite(x)

    # def loss_x_y(self, x: np.ndarray, y: np.ndarray):
    #     return self.loss(y, self(x))

    def print_gradients(self, x, y):
        for f in self.functions:
            display(self.get_gradient(f, x, y))

    def get_gradient(self, f_base, x, y):
        """
        hier lässt sich noch viel rausholen, da die multiplikation im prinzip iterativ ist und der hintere teil immer wieder verwendet wird.
        """
        # def get_gradient(self, f2, a):
        # functions_after = [f2, f3, f4]
        # return f2.paramsgradient(a) @ f3.inputgradient(f2(a)) @ f4.inputgradient(f3(f2(a)))

        # return reduce(@, [f.right_gradient(reduce(compose, fs_before)(a)) for f in functions_after])

        functions_after = self.functions[self.functions.index(f_base) + 1:]

        def correct_gradient(f, f_base):
            if f == f_base:
                return f.params_gradient
            else:
                return f.input_gradient

        gradients = [
            (correct_gradient(f, f_base))(
                reduce(compose, self.functions[: self.functions.index(f)])(x)
            )
            for f in functions_after
        ]
        return reduce(np.matlul, gradients) @ self.loss.input_gradient(y, self(x))
