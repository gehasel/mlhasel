import numpy as np
from typing import List, Optional
from functools import reduce
from IPython.display import display
import logging

np.set_printoptions(suppress=True, precision=4)


class NamedObject:
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return super().__str__()


class MLFunction(NamedObject):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)

    def __str__(self) -> str:
        if self.name is not None:
            return self.name
        else:
            return super().__str__()

    @property
    def weight_vector(self) -> np.ndarray:
        raise NotImplementedError

    @weight_vector.setter
    def weight_vector(self, stacked_weight_vector: np.ndarray) -> None:
        """
        Note: the order in which the parameters are set (and returned)
        has to coincide with the order of the derivatives
        in the rows of the params_gradient.
        (In each row there are the derivatives for one of the parameters)
    
        """
        raise NotImplementedError

    def __call__(self, a: np.ndarray) -> np.ndarray:
        """
        forward pass

        a:  column vector of shape (input_dim, 1) which is the output of the previous mlfunction,
            if this is the first mlfunction, a == x
        """
        raise NotImplementedError

    def params_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: output of the previous mlfunction in forward pass
        """
        raise NotImplementedError

    def input_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: output of the previous mlfunction in forward pass
        """
        raise NotImplementedError


class Lin(MLFunction):
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

    @property
    def weight_vector(self):
        """
        Return a column vector of shape (number of weights, 1)
        """
        return np.concatenate((self.weight_matrix.T.flatten(), self.bias_vector.T.flatten())).reshape(-1, 1)

    @weight_vector.setter
    def weight_vector(self, stacked_weight_vector: np.ndarray):
        """
        stacked_weight_vector: column vector of shape (number of weights, 1)
        """
        matrix_portion = stacked_weight_vector[0:self.weight_matrix.size]
        bias_portion = stacked_weight_vector[self.weight_matrix.size:]

        n, m = self.weight_matrix.shape
        # note how m and n are switched here:
        self.weight_matrix = matrix_portion.reshape(m, n).T
        self.bias_vector = bias_portion.reshape(-1, 1)

    def __call__(self, a: np.ndarray):
        """
        a is a column vector, if this is the first mlfunction, a == x
        """
        assert a.shape == (self.input_dim, 1)
        r = (self.weight_matrix.T @ a) + self.bias_vector
        assert r.shape == (self.output_dim, 1)
        return r

    def params_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: input of the forward pass at this mlfunction
        """
        assert a.shape == (self.input_dim, 1)
        upper_matrix = np.zeros((self.input_dim * self.output_dim, self.output_dim))
        for i in range(self.output_dim):
            # Fill the ith column with the vector a:
            upper_matrix[i * self.input_dim: (i + 1) * self.input_dim, i] = a[:, 0]
        lower_matrix = np.eye(self.output_dim)  # grad of the bias terms
        result_matrix = np.vstack((upper_matrix, lower_matrix))
        assert result_matrix.shape == (self.num_params, self.output_dim)
        return result_matrix

    def input_gradient(self, a: np.ndarray) -> np.ndarray:
        """
        a: input of the forward pass at this mlfunction
        """
        # print(f"{a.shape=}")
        # print(f"{self.input_dim=}")
        assert a.shape == (self.input_dim, 1)
        result_matrix = self.weight_matrix
        assert result_matrix.shape == (self.input_dim, self.output_dim)
        return result_matrix


class Softmax(MLFunction):
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

    @property
    def weight_vector(self) -> np.ndarray:
        return np.empty((0, 1))

    @weight_vector.setter
    def weight_vector(self, flat_vector: np.ndarray) -> None:
        pass


class ReLu(MLFunction):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, a: np.ndarray):
        assert a.shape[1] == 1
        return a * (a > 0)

    def input_gradient(self, a: np.ndarray):
        assert a.shape[1] == 1
        dim = a.shape[0]
        J = np.zeros((dim, dim))
        np.fill_diagonal(J, a > 0)
        r = J.T
        return r

    def params_gradient(self, a: np.ndarray) -> np.ndarray:
        assert a.shape[1] == 1
        return np.empty((0, a.shape[0]))  # returns a matrix with no rows

    @property
    def weight_vector(self) -> np.ndarray:
        return np.empty((0, 1))

    @weight_vector.setter
    def weight_vector(self, flat_vector: np.ndarray) -> None:
        pass


class Loss(NamedObject):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def __call__(self, y: np.ndarray, y_pred: np.ndarray):
        """
        Return a real value.

        y_pred == f_z_L == last_activation_function(z_L)
        """
        raise NotImplementedError

    def input_gradient(self, y: np.ndarray, y_pred: np.ndarray):
        """
        The gradient with respect to the input (= the output of the last mlfunction)
        """
        raise NotImplementedError


class CE(Loss):
    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def __call__(self, y: np.ndarray, y_pred: np.ndarray):
        matrix1x1 = -(y.T @ np.log(y_pred))
        flattened_to_float = matrix1x1.item()
        return flattened_to_float

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


def compose(f, g):
    return lambda x: f(g(x))


def identity(e):
    return e


class Composition(MLFunction):
    def __init__(self, ml_functions: List[MLFunction]):
        self.ml_functions = ml_functions
        self.composite = reduce(
            compose, reversed(self.ml_functions)
        )  # like foldl or foldr (both, since composition is associative)

    def __call__(self, x: np.ndarray):
        return self.composite(x)

    def input_gradient(self, x: np.ndarray) -> np.ndarray:
        gradients = [
            (f.input_gradient)(
                reduce(
                    compose,
                    reversed(self.ml_functions[: self.ml_functions.index(f)]),
                    identity,
                )(x)
            )
            for f in self.ml_functions
        ]
        return reduce(np.matmul, gradients)

    def f_base_params_gradient(self, f_base: callable, x: np.ndarray) -> np.ndarray:
        """
        This has lots of potential for optimization, since in the computation of the individual gradients
        there is always a muultiplication with the previous one.
        This is a very iterative process that can be greatly sped up, but then it would not show the
        chain rule as neatly.
        """
        # What happens when calling e.g. f_base_params_gradient(self, f2, x, y)?

        # functions_after = [f2, f3, f4]
        functions_from_f_base_on = self.ml_functions[self.ml_functions.index(f_base) :]

        def correct_gradient(f, f_base):
            if f == f_base:
                return f.params_gradient
            else:
                return f.input_gradient

        # == [f2.params_gradient(reduce(compose, [f_1])(x)), f_3.input_gradient(reduce(compose, [f2, f_1])(x)), f_4.input_gradient(reduce(compose, [f_3, f2, f_1])(x))]
        # == [f2.params_gradient(f_1(x)), f_3.input_gradient(f2(f_1(x))), f_4.input_gradient(f_3(f2(f_1(x))))]
        gradients = [
            (correct_gradient(f, f_base))(
                reduce(compose, reversed(self.ml_functions[: self.ml_functions.index(f)]), identity)(x)
            )
            for f in functions_from_f_base_on
        ]

        # return f2.paramsgradient(f_1(x)) @ f3.inputgradient(f2(f_1(x))) @ f4.inputgradient(f3(f2(f_1(x))))
        return reduce(np.matmul, gradients)

    def params_gradient_naive(self, x: np.ndarray) -> np.ndarray:
        return np.vstack([self.f_base_params_gradient(f, x) for f in self.ml_functions])

    @property
    def weight_vector(self) -> np.ndarray:
        return np.vstack([f.weight_vector for f in self.ml_functions])

    @weight_vector.setter
    def weight_vector(self, stacked_weight_vector: np.ndarray) -> None:
        start = 0
        for function in self.ml_functions:
            n = function.weight_vector.shape[0]  # Size of the individual weight vector
            end = start + n
            function.weight_vector = stacked_weight_vector[start:end]
            start = end

    def print_gradients(self, x, y):
        for f in self.ml_functions:
            print(f"Gradient of {f}:")
            grad = self.params_gradient_f_base(f, x, y)
            print(f"{grad.shape=}")
            print(f"{f.weight_vector.shape=}")
            display(grad)


def to_column_vector(v: np.ndarray) -> np.ndarray:
    return v.reshape(v.shape[0], 1)


class Optimizer(NamedObject):
    def optimize(model: 'Model'):
        raise NotImplementedError


class BatchGradientDescent(Optimizer):
    def __init__(self, batch_size, learning_rate: float, epochs: int):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def optimize(self, model: 'Model'):
        for i in range(self.epochs):
            logging.debug(f"entered epoch {i}")
            batch_count = 0
            idx_low = self.batch_size * batch_count  # 0
            idx_high = self.batch_size * (batch_count + 1)  # 12

            # I believe this loop cannot be parallelized, since it depends on the weights from the previous iteration
            while idx_low < (model.X_train.shape[0]):  # 0 <= 12
                logging.debug(f"entered batch {batch_count}")
                X_batch = model.X_train[idx_low: idx_high]
                Y_batch = model.Y_train[idx_low: idx_high]

                # this is just a for loop around getting the gradients, can this be paralellized?
                mutual_grad_sum = np.zeros(model.composition.weight_vector.shape)
                batch_loss_sum = 0
                for k, (x, y) in enumerate(zip(X_batch, Y_batch)):
                    logging.debug(f"eval point {k}")
                    x_cv = to_column_vector(x)
                    y_cv = to_column_vector(y)
                    mutual_grad_sum += model.loss_params_gradient(x_cv, y_cv)
                    batch_loss_sum += model.loss_x_y(x_cv, y_cv)
                mutual_grad = mutual_grad_sum / X_batch.shape[0]
                batch_loss = batch_loss_sum / X_batch.shape[0]
                model.composition.weight_vector -= self.learning_rate * mutual_grad

                logging.info(f"average loss in batch {batch_count} before its parameter update: {batch_loss}")

                batch_count += 1
                idx_low = self.batch_size * batch_count
                idx_high = self.batch_size * (batch_count + 1)
        logging.info(f"epoch {i} done.")


class Model(NamedObject):
    def __init__(
        self,
        ml_functions: List[MLFunction],
        loss: Loss,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        optimizer: Optional[Optimizer] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        a Model is fully determined after initiation,
        the only things that still happen are the training procedure and prediction

        The intent was to keep this as functional as possible, while still separating model definition from computation
        To iterate on an existing model with different training data or optimizer, a new model has to be created
        that uses the ml_functions of the old model.

        What are the differences in responsibility for composition and model?
        model: implements convenience functions around its composition, takes care of input requirements of composition
        composition: requires column vectors, able to be used as a function in other compositions
        """
        super().__init__(name)
        self.ml_functions = ml_functions
        self.composition = Composition(self.ml_functions)
        self.loss = loss
        self.X_train = X_train  # shape = (number of samples, number of features)
        self.Y_train = Y_train  # shape = (number of samples, number of output dimensions)
        self.optimizer = optimizer
        self.name = name
        self.trained = False

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_cv = to_column_vector(x)
        return self.composition(x_cv)

    def loss_x_y(self, x: np.ndarray, y: np.ndarray):
        x_cv = to_column_vector(x)
        y_cv = to_column_vector(y)
        return self.loss(y_cv, self.predict(x_cv))

    def loss_params_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_cv = to_column_vector(x)
        y_cv = to_column_vector(y)
        return self.composition.params_gradient(x) @ self.loss.input_gradient(y_cv, self.composition(x_cv))

    def fit(self):
        self.optimizer.optimize(self)
        self.trained = True
