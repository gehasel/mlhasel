# %%
import numpy as np
from IPython.display import display

from gradient import Lin, Softmax, CE, Composition, compose, Model

np.set_printoptions(suppress=True)

# %%
rng = np.random.default_rng(seed=123)

x = rng.integers(low=0, high=10, size=(10, 1))

z_1 = Lin(rng.random(size=(10, 5)), rng.random(size=(5,1)), name='z_1')
f_1 = Softmax(name='f_1')
z_2 = Lin(rng.random(size=(5, 3)), rng.random(size=(3,1)), name='z_2')
f_2 = Softmax(name='f_2')

# if entries of y sum to 1, then the grad
# of the loss + softmax = -(y-y_pred)
y = np.array([[0.3], [0.2], [0.5]])  # rng.random(size=3)
loss = CE(name='loss')

# %%
display(x)
# %%
y_pred = f_2(z_2(f_1(z_1(x))))
display(y_pred)
# %%
display(loss(y, y_pred))

# %%
nn = Composition([z_1, f_1, z_2, f_2])
nn(x)
# %%
mymodel = Model(nn, loss)
mymodel.loss(y, y_pred)
# %%
np.matmul(f_2.input_gradient(z_2(f_1(z_1(x)))), loss.input_gradient(y, y_pred))
# %%
# should equal the above
-(y - y_pred*y_pred.sum(axis=0))
# %%
grad_z2 = z_2.params_gradient(f_1(z_1(x))) @ f_2.input_gradient(z_2(f_1(z_1(x)))) @ loss.input_gradient(y, y_pred)
display(grad_z2)
print(f"{grad_z2.shape=}")
print(f"{z_2.weight_vector.shape=}")
# %%
grad_z1 = z_1.params_gradient(x) @ f_1.input_gradient(z_1(x)) @ z_2.input_gradient(f_1(z_1(x))) @ f_2.input_gradient(z_2(f_1(z_1(x)))) @ loss.input_gradient(y, y_pred)
display(grad_z1)
print(f"{grad_z1.shape=}")
print(f"{z_1.weight_vector.shape=}")
# %%
from functools import reduce
reduce(compose, [z_1])(x)
# %%
mymodel.loss_params_gradient(x, y)
# %%
mymodel.loss_x_y(x, y)

# %%
mymodel.composition.weight_vector.shape

# %%
mymodel.loss_params_gradient(x, y).shape
# %%
print(mymodel.composition.ml_functions[0].weight_matrix.shape)
print(mymodel.composition.ml_functions[2].weight_matrix.shape)
# %%
mymodel.composition.weight_vector = mymodel.composition.weight_vector - 0.001 * mymodel.loss_params_gradient(x, y)

#%%
mymodel.loss_x_y(x, y)
