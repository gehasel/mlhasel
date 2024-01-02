# %%
import numpy as np
from IPython.display import display

from gradient import Lin, Softmax, CE, NN, compose

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
nn = NN([z_1, f_1, z_2, f_2], loss)
nn(x)
# %%
nn.loss(y, y_pred)
# %%
np.matmul(f_2.input_gradient(z_2(f_1(z_1(x)))), loss.input_gradient(y, y_pred))
# %%
# should equal the above
-(y - y_pred*y_pred.sum(axis=0))
# %%
grad_z2 = z_2.params_gradient(f_1(z_1(x))) @ f_2.input_gradient(z_2(f_1(z_1(x)))) @ loss.input_gradient(y, y_pred)
display(grad_z2)
print(f"{grad_z2.shape=}")
print(f"{z_2.get_weight_vector().shape=}")
# %%
grad_z1 = z_1.params_gradient(x) @ f_1.input_gradient(z_1(x)) @ z_2.input_gradient(f_1(z_1(x))) @ f_2.input_gradient(z_2(f_1(z_1(x)))) @ loss.input_gradient(y, y_pred)
display(grad_z1)
print(f"{grad_z1.shape=}")
print(f"{z_1.get_weight_vector().shape=}")
# %%
from functools import reduce
reduce(compose, [z_1])(x)
# %%
nn.print_gradients(x, y)
# %%
x = np.array([[1], [2]])
y = np.array([[2], [3]])

z_1_prime = Lin(np.array([[0, 3], [2, 1]]), np.array([[1], [2]]))
f_1_prime = Softmax()
loss2 = CE()

nn2 = NN([z_1_prime, f_1_prime], loss2)
y_pred1 = f_1_prime(z_1_prime(x))
y_pred2 = nn2(x)
print('Correct output?')
display(all(y_pred1 == y_pred2))

print('prediction:')
display(y_pred1)

print('loss:')
display(nn2.loss(x, y))


# %%
np.matmul(f_1_prime.gradient(z_1_prime(x)), loss2.gradient(y, y_pred1))
# %%
-(y - y_pred1*y.sum())
# %%
z_1_prime(x)
# %%
f_1_prime.gradient(z_1_prime(x))

# %%
loss2.gradient(y, y_pred1)
# %%
np.matmul(np.array([[ 0.10499359, -0.10499359], [-0.10499359, 0.10499359]]), loss2.gradient(y, y_pred1))
# %%
