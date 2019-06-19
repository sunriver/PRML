import numpy as np
from scipy.stats import truncnorm
from prml.nn.array.array import asarray


def normal(mean, std, size):
    return asarray(np.random.normal(mean, std, size))


def truncnormal(min, max, scale, size):
    #https://www.zhihu.com/question/49923924 截断正态分布
    return asarray(truncnorm(a=min, b=max, scale=scale).rvs(size))
