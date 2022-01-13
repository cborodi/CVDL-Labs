import numpy as np

def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    # print(x)
    x = x * 0.0000001
    # print(x.max())
    exp_x = np.exp(x/t)
    sum_exp_x = np.sum(exp_x, axis=1)
    sm_x = exp_x / sum_exp_x[:, None]
    return sm_x
