# %%
import mnist_for_numpy.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from gradient import Model, GradientDescent, CE, Lin, Softmax, to_column_vector, ReLu

np.set_printoptions(suppress=True, precision=4)
# %%
X_train, Y_train, X_test, Y_test = mnist.load()
# %%
X_train = X_train[:1000] / 255.
X_test = X_test / 255.

num = 1
x = X_train[num, :]
y = Y_train[num]

img = x.reshape(28, 28) # First image in the training set.
plt.imshow(img, cmap='gray')
plt.show()  # Show the image
print(f"actual: {y}")
# %%


def one_hot_encode(arr):
    one_hot = np.zeros((arr.size, 10))
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


Y_train_ohe = one_hot_encode(Y_train)
Y_test_ohe = one_hot_encode(Y_test)

print(f"{X_train.shape=}")
print(f"{Y_train_ohe.shape=}")
# %%
display(X_train[:5, :])
display(Y_train_ohe[:5])

x_cv = to_column_vector(x)
y_cv = to_column_vector(Y_train_ohe[num])
# %%

rng = np.random.default_rng(seed=123)

lower_rand = -1
upper_rand = 1
in1 = 784
out1 = 300
z_1 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(in1, out1)), rng.uniform(low=lower_rand, high=upper_rand, size=(out1, 1)), name='z_1')
f_1 = ReLu(name='f_1')

out2 = 100
z_2 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(out1, out2)), rng.uniform(low=lower_rand, high=upper_rand, size=(out2, 1)), name='z_2')
f_2 = ReLu(name='f_2')

out3 = 100
z_3 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(out2, out3)), rng.uniform(low=lower_rand, high=upper_rand, size=(out3, 1)), name='z_3')
f_3 = ReLu(name='f_3')

out4 = 10
z_4 = Lin(rng.uniform(low=lower_rand, high=upper_rand, size=(out3, out4)), rng.uniform(low=lower_rand, high=upper_rand, size=(out4, 1)), name='z_4')
f_4 = Softmax(name='f_4')

loss = CE()

myoptimizer = GradientDescent(0.005, 10)
mymodel = Model(
    [z_1, f_1, z_2, f_2, z_4, f_4],
    loss,
    X_train,
    Y_train_ohe,
    myoptimizer,
)
# %%
mymodel.predict(x)
# %%
# print(z_1(x_cv))
# print(f_1(z_1(x_cv)))
# print(z_2(f_1(z_1(x_cv))))
# print(f_2(z_2(f_1(z_1(x_cv)))))
# print(z_3(f_2(z_2(f_1(z_1(x_cv))))))
# print(f_3(z_3(f_2(z_2(f_1(z_1(x_cv)))))))
# print(z_4(f_3(z_3(f_2(z_2(f_1(z_1(x_cv))))))))
# print(f_4(z_4(f_3(z_3(f_2(z_2(f_1(z_1(x_cv)))))))))

# print(z_1(x_cv))
# print(f_1(z_1(x_cv)))
# print(z_4(f_1(z_1(x_cv))))
# print(f_4(z_4(f_1(z_1(x_cv)))))
# %%
np.savetxt('array.csv', (mymodel.loss_params_gradient(x_cv, y_cv)), delimiter=',')

# %%
mymodel.composition.weight_vector.shape
# %%
mymodel.composition.weight_vector -= 0.05 * (mymodel.composition.params_gradient(x_cv) @ mymodel.loss.input_gradient(y_cv, mymodel.predict(x)))
mymodel.loss_x_y(x_cv, y_cv)
# %%
mymodel.fit()

# %%
np.min(x_cv)
# %%
