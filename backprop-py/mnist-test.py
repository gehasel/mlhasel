# %%
import mnist_for_numpy.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import logging

from gradient import Model, BatchGradientDescent, CE, Lin, Softmax, to_column_vector, ReLu

# %%
np.set_printoptions(suppress=True, precision=4)
# %%

logging.basicConfig(level=logging.DEBUG)

# %%
X_train, Y_train, X_test, Y_test = mnist.load()
# %%
num_datapoints = 1000
X_train = X_train[:num_datapoints] / 255.
X_test = X_test / 255.

num = 6
x = X_train[num, :]
y = Y_train[num]

# %%
def show_img(x):
    img = x.reshape(28, 28) # First image in the training set.
    plt.imshow(img, cmap='gray')
    plt.show()  # Show the image
# %%

show_img(x)
print(f"actual: {y}")
# %%


def one_hot_encode(arr):
    one_hot = np.zeros((arr.size, 10))
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


Y_train_ohe = one_hot_encode(Y_train[:num_datapoints])
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

lower_rand = -0.1
upper_rand = 0.1
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

myoptimizer = BatchGradientDescent(10, 0.1, 1)
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
# np.savetxt('array.csv', (mymodel.loss_params_gradient(x_cv, y_cv)), delimiter=',')

# %%
# mymodel.composition.weight_vector.shape
# # %%
# mymodel.composition.weight_vector -= 0.05 * (mymodel.composition.params_gradient(x_cv) @ mymodel.loss.input_gradient(y_cv, mymodel.predict(x)))
# print(mymodel.predict(x_cv))
# mymodel.loss_x_y(x_cv, y_cv)

# %%
mymodel.fit()

# %%
# np.min(x_cv)
# # %%
# print(mymodel.predict(x_cv))
# # %%
for test_num in range(5):
    x_test1 = X_test[test_num]
    show_img(x_test1)
    print(f"{np.argmax(mymodel.predict(x_test1))} with confidence {np.max(mymodel.predict(x_test1)):.4f}")
    print(f"second best guess: {np.argsort(np.max(mymodel.predict(x_test1), axis=1))[-2]} with confidence {np.sort(np.max(mymodel.predict(x_test1), axis=1))[-2]:.4f}")
    print(f"actual: {Y_test[test_num]}")
# %%
