import numpy as np
from prml.nn.array.array import Array, asarray
from prml.nn.config import config
from prml.nn.queue import backprop_queue


class Function(object):
    enable_auto_broadcast = False

    #构建计算图, 输出为out array, 持有操作符self, self持有输入args 和 kwargs,  
    def forward(self, *args, **kwargs):
        self.args = [self._convert2array(arg) for arg in args]
        if self.enable_auto_broadcast:
            self.args = self._autobroadcast(self.args)
        self.kwargs = kwargs
        out = self._forward(*tuple(arg.value for arg in self.args), **kwargs)
        if config.enable_backprop:
            return Array(out, parent=self)
        else:
            return Array(out, parent=None)

    #把计算图的输出out反向压入队列，准备为后续更新梯度和权值, delta为op节点输出误差, sefl.args, kwargs为输入项
    def backward(self, delta):
        #计算所有输入参数的变化量dargs，对应参数梯度
        dargs = self._backward(delta, *tuple(arg.value for arg in self.args), **self.kwargs)
        if isinstance(dargs, tuple):
            for arg, darg in zip(self.args, dargs):
                arg._backward(darg)
                #由于queue存放的是输出 array，由于一个输出可能有多个输入参数args，所以为防止array多次压入queue，
                # 因此设置flag ‘is_in_queue’
                if not arg.is_in_queue:
                    backprop_queue.enqueue(arg)
        else:
            self.args[0]._backward(dargs)
            if not self.args[0].is_in_queue:
                backprop_queue.enqueue(self.args[0])

    def _out_depth(self):
        return max([arg.depth for arg in self.args]) + 1

    @staticmethod
    def _autobroadcast(arg):
        return broadcast(arg)

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _backward(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _convert2array(arg):
        if not isinstance(arg, Array):
            return asarray(arg)
        else:
            return arg


class BroadcastTo(Function):
    """
    Broadcast a tensor to an new shape
    """

    def __init__(self, shape):
        self.shape = shape

    def _forward(self, x):
        output = np.broadcast_to(x, self.shape)
        return output

    @staticmethod
    def _backward(delta, x):
        dx = delta
        xdim = getattr(x, "ndim", 0)
        xshape = getattr(x, "shape", ())
        if delta.ndim != xdim:
            dx = dx.sum(axis=tuple(range(dx.ndim - xdim)))
            if isinstance(dx, np.number):
                dx = np.array(dx)
        axis = tuple(i for i, len_ in enumerate(xshape) if len_ == 1)
        if axis:
            dx = dx.sum(axis=axis, keepdims=True)
        return dx


def broadcast_to(x, shape):
    """
    Broadcast a tensor to an new shape
    """
    return BroadcastTo(shape).forward(x)


def broadcast(args):
    """
    broadcast list of tensors to make them have the same shape

    Parameters
    ----------
    args : list
        list of Tensor to be aligned

    Returns
    -------
    list
        list of Tensor whose shapes are aligned
    """
    shape = np.broadcast(*(arg.value for arg in args)).shape
    for i, arg in enumerate(args):
        if arg.shape != shape:
            args[i] = BroadcastTo(shape).forward(arg)
    return args
