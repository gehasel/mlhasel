import numpy as np
from typing import List, Optional
from functools import reduce
from IPython.display import display


class NamedObject:
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return super().__str__()

    def __repr__(self) -> str:
        return self.__str__()


class Layer(NamedObject):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return super().__str__()

    def __repr__(self) -> str:
        return self.__str__()

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
    def __init__(
        self,
        weight_matrix: np.ndarray,
        bias_vector: np.ndarray,
        name: Optional[str] = None,
    ):
        """
        weight_matrix of shape (number of inputs, number of outputs)
        bias_vector of shape (number of outputs, 1)
        """
        super().__init__(name=name)
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
            upper_matrix[i * self.input_dim : (i + 1) * self.input_dim, i] = a[:, 0]
        lower_matrix = np.eye(self.output_dim)  # grad of the bias terms
        result_matrix = np.vstack((upper_matrix, lower_matrix))
        assert result_matrix.shape == (self.num_params, self.output_dim)
        return result_matrix

    def input_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: input of the forward pass at this layer
        """
        # print(f"{a.shape=}")
        # print(f"{self.input_dim=}")
        assert a.shape == (self.input_dim, 1)
        result_matrix = self.weight_matrix
        assert result_matrix.shape == (self.input_dim, self.output_dim)
        return result_matrix


class Softmax(Layer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, a: np.ndarray):
        assert a.shape[1] == 1
        a_max = np.max(a, axis=0)
        a_norm = a - a_max
        r = np.divide(np.exp(a_norm), np.sum(np.exp(a_norm), axis=0))
        return r

    def input_gradient(self, a: np.ndarray):
        """
        The transpose of the jacobian where every derivative in the jacobian
        evaluated a
        """
        assert a.shape[1] == 1
        exp_z = np.exp(a)
        normalized_exp_z = exp_z / np.sum(exp_z)
        J = -np.outer(normalized_exp_z, normalized_exp_z)
        # for some reason, we must flatten normalized... here,
        # otherwise it only adds the first element of b to the diagonal, not elementwise
        np.fill_diagonal(J, normalized_exp_z.flatten() + J.diagonal())
        # J is now the jacobian where every partial derivative has already evaluated a
        # Although symmetric, I transpose for consistency
        r = J.T
        return r

    def params_gradient(self, a: np.ndarray) -> np.ndarray:
        assert a.shape[1] == 1
        return np.empty((0, a.shape[0]))  # returns a matrix with no rows

    def get_weight_vector(self) -> np.ndarray:
        return np.empty((0, 1))

    def set_weight_vector(self, flat_vector: np.ndarray) -> None:
        pass


class CE(Layer):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

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

    def get_weight_vector(self) -> np.ndarray:
        return np.empty((0, 1))

    def set_weight_vector(self, flat_vector: np.ndarray) -> None:
        pass


def compose(f, g):
    return lambda x: f(g(x))


def identity(e):
    return e


class Composition(Layer):
    def __init__(self, functions: List[callable]):
        self.functions = functions
        self.composite = reduce(
            compose, reversed(functions)
        )  # like foldl or foldr (both, since composition is associative)

    def __call__(self, x: np.ndarray):
        return self.composite(x)

    def input_gradient(self, x: np.ndarray) -> np.ndarray:
        gradients = [
            (f.input_gradient)(
                reduce(
                    compose,
                    reversed(self.functions[: self.functions.index(f)]),
                    identity,
                )(x)
            )
            for f in self.functions
        ]
        return reduce(np.matmul, gradients)

    def f_base_params_gradient(self, f_base: callable, x: np.ndarray) -> np.ndarray:
        """
        hier lässt sich noch viel rausholen, da die multiplikation im prinzip iterativ ist und der hintere teil immer wieder verwendet wird.
        """
        # def get_gradient(self, f2, x, y):
        # functions_after = [f3, f4]
        # return f2.paramsgradient(a) @ f3.inputgradient(f2(a)) @ f4.inputgradient(f3(f2(a)))

        # return reduce(@, [f.right_gradient(reduce(compose, fs_before)(a)) for f in functions_after])

        functions_after = self.functions[self.functions.index(f_base)+1:]  # [f2, f3, f4]

        gradients = [
            (f.input_gradient)(
                reduce(
                    compose,
                    reversed(self.functions[: self.functions.index(f)]),
                    identity,
                )(x)
            )
            for f in functions_after
        ]  # [f_2.params_gradient(reduce(compose, [f_1])(x)), f_3.input_gradient(reduce(compose, [f_2, f_1])(x)), f_4.input_gradient(reduce(compose, [f_3, f_2, f_1])(x))]
        # == [f_2.params_gradient(f_1(x)), f_3.input_gradient(f_2(f_1(x))), f_4.input_gradient(f_3(f_2(f_1(x))))]

        # gradients = []
        # for f in functions_after:
        #     inner_functions = self.functions[:self.functions.index(f)]
        #     inner_functions.reverse()
        #     # print(list(inner_functions))
        #     inner_eval = reduce(compose, inner_functions, identity)(x)
        #     correct_gradient_f = correct_gradient(f, f_base)
        #     gradients.append(correct_gradient_f(inner_eval))
        return f_base.params_gradient(x) @ reduce(np.matmul, gradients)

    def params_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([self.f_base_params_gradient(f, x) for f in self.functions])

    def get_weight_vector(self) -> np.ndarray:
        raise NotImplementedError

    def set_weight_vector(self, stacked_weight_vector: np.ndarray) -> None:
        raise NotImplementedError

    def print_gradients(self, x, y):
        for f in self.functions:
            print(f"Gradient of {f}:")
            grad = self.params_gradient_f_base(f, x, y)
            print(f"{grad.shape=}")
            print(f"{f.get_weight_vector().shape=}")
            display(grad)


class Model(NamedObject):
    def __init__(self, composition: Composition, loss: callable, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.composition = composition
        self.loss = loss
        self.name = name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.composition(x)

    def loss_x_y(self, x: np.ndarray, y: np.ndarray):
        return self.loss(y, self(x))

    def loss_params_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.composition.params_gradient(x) @ self.loss.input_gradient(y, self(x))
