import numpy as np
from activations import softmax

class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # TODO your code here
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find np.random.randn userful here *0.001
        self.W = 0.001 * np.random.randn(self.input_shape, self.num_classes)
        # self.W = np.around(self.W, 4)
        # self.load("weights.txt")
        # print(self.W)
        # print(self.W.shape)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X
        # remember about the bias trick!
        # 1. apply the softmax function on the scores
        # 2, returned the normalized scores
        # bias = np.ones([1, 1])
        # X = np.concatenate([X, bias], 0)
        pred = np.dot(X, self.W)
        scores = softmax(pred)
        return scores

    def predict(self, X: np.ndarray):
        label = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X as the scores
        # 1. compute the prediction by taking the argmax of the class scores
        # bias = np.ones([1, 1])
        # X = np.concatenate([X, bias], 0)
        pred = np.dot(X, self.W)
        scores = softmax(pred)
        label = np.argmax(scores, axis=1)
        return label

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        steps = kwargs['steps'] if 'steps' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print(bs, reg_strength, steps, lr)

        # run mini-batch gradient descent
        for iteration in range(0, steps):
            # TODO your code here
            # sample a batch of images from the training set
            # you might find np.random.choice useful
            indices = np.random.choice(10000, bs, replace=False)
            X_batch, y_batch = X_train[indices], y_train[indices]
            # print(X_batch)
            # print(y_batch)
            groundTruth = np.zeros(shape=(bs, 10)) # 10 -> possible outputs
            aux_indices = np.arange(bs)
            groundTruth[aux_indices, y_batch] = 1
            CT = np.subtract(softmax(np.dot(X_batch, self.W)), groundTruth)
            dW = np.dot(X_batch.T, CT)

            self.W -= lr * dW

            exp_x = np.exp(np.dot(X_batch, self.W) * 0.0000001)
            sum_exp_x = np.sum(exp_x, axis=1)
            kk = np.arange(bs)
            initial_loss = np.add(-(np.dot(X_batch, self.W) * 0.0000001)[kk, y_batch], np.log(sum_exp_x))

            loss = (1 / bs) * np.sum(initial_loss) # + reg_strength * np.sum(np.square(self.W))
            # end TODO your code here
            # compute the loss and dW
            # perform a parameter update

            # self.W = np.around(self.W, 2)
            # append the training loss, accuracy on the training set and accuracy on the test set to the history dict
            history.append(loss)

        return history


    def get_weights(self, img_shape):
        W = None
        # TODO your code here
        # 0. ignore the bias term
        # 1. reshape the weights to (*image_shape, num_classes)
        W = self.W[:-1][:]
        W = W.reshape((img_shape[0], img_shape[1], img_shape[2], 10))
        return W

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        lines = []
        with open(path) as f:
            lines = f.readlines()

        shape = int(lines[0])
        self.input_shape = shape

        numberOfClasses = int(lines[1])

        self.W = np.empty([shape, numberOfClasses])
        for line in range(2, len(lines)):
            weights = lines[line]
            weights = weights.split(" ")
            for i in range(len(weights)):
                self.W[line - 2][i] = float(weights[i])

        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find np.save useful for this
        # TODO your code here
        f = open(path, "w")
        f.write(str(self.input_shape))
        f.write('\n')
        f.write(str(self.num_classes))
        f.write('\n')
        np.savetxt(path, self.W, fmt='%f')
        return True

"""
x = SoftmaxClassifier((1, 3072), 10)
x.save('weights.txt')
"""

"""
print(np.random.choice(50000, 128, replace=False)) # indices
after defining X_train, y_train locally
X_batch, y_batch = X_train[indices], y_train[indices]
"""

"""
groundTruth = np.zeros(shape=(10, 10)) # 10 -> possible outputs
aux_indices = np.arange(10)
indices = np.array([4, 1, 2, 3, 4, 8, 7, 9, 1, 7])
groundTruth[aux_indices, indices] = 1
print(groundTruth)
"""

"""
x = np.array([1, 2, 2])
t = 1
exp_x = np.exp(x/t)
print(exp_x)
sum_exp_x = np.sum(exp_x, axis=1)
print(sum_exp_x)
# sm_x = exp_x / sum_exp_x
# sm_x = np.divide(exp_x.T, sum_exp_x).T
sm_x = exp_x / sum_exp_x[:, None]
print(sm_x)
"""


from lab2 import cifar10

cifar_root_dir = 'cifar-10-batches-py'
X_train, y_train, X_test, y_test = cifar10.load_ciaf10(cifar_root_dir)
indices = np.random.choice(len(X_train), 10000, replace=False)

X_train = X_train.astype(np.float32)

X_train = np.reshape(X_train, (X_train.shape[0], -1))

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

display_images, display_labels = X_train[indices], y_train[indices]

from functools import reduce
import random

input_size_flattened = reduce((lambda a, b: a * b), display_images[0].shape)

lr_bounds = (-7, -2)
reg_strength_bounds = (-4, -2)

learning_rates = [-1, -1]
regularization_strengths = [3000, 80000]

lr = pow(10, random.uniform(learning_rates[0], learning_rates[1]))
reg_strength = random.uniform(regularization_strengths[0], regularization_strengths[1])


cls = SoftmaxClassifier(input_shape=input_size_flattened, num_classes=cifar10.NUM_CLASSES)

history = cls.fit(display_images, display_labels, lr=lr, reg_strength=reg_strength,
        steps=5000, bs=256)

y_train_pred = cls.predict(display_images)

train_acc = np.mean(display_labels == y_train_pred)

print('\rlr {:.10f}, reg_strength {:.2f}, train_acc {:.2f}\n'.format(lr, reg_strength, train_acc))

print(history)