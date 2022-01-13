# validate softmax
import numpy as np
# import tensorflow as tf
from activations import softmax
import matplotlib.pyplot as plt

x = np.asarray([20, 30, -15, 45, 39, -10])
x1 = np.asarray([[20, 30, -15, 45, 39, -10]])
T = [0.25, 0.75, 1, 1.5, 2, 5, 10, 20, 30]
x.sort()

plt.rcParams["figure.figsize"] = (15, 3)

for idx in range(0, len(T)):
    y = softmax(x1, T[idx])
    plt.subplot(1, len(T), idx + 1)
    print(x1)
    y1 = np.array(y[0])
    print(y)
    plt.plot(x, y1)
    plt.grid(True)
    plt.title(T[idx])

plt.subplots_adjust(wspace=1)
plt.show()
