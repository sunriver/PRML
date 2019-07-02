import numpy as np
from prml.kernel.kernel import Kernel


class RBF(Kernel):

    def __init__(self, params):
        """
        construct Radial basis kernel function

        Parameters
        ----------
        params : (ndim + 1,) ndarray
            parameters of radial basis function

        Attributes
        ----------
        ndim : int
            dimension of expected input data
        """
        assert params.ndim == 1
        self.params = params
        self.ndim = len(params) - 1

    def __call__(self, x, y, pairwise=True):
        """
        calculate radial basis function
        k(x, y) = c0 * exp(-0.5 * c1 * (x1 - y1) ** 2 ...)

        Parameters
        ----------
        x : ndarray [..., ndim]
            input of this kernel function
        y : ndarray [..., ndim]
            another input

        Returns
        -------
        output : ndarray
            output of this radial basis function
        """

        #The shape attribute for numpy arrays returns the dimensions of the array. 
        # If Arr has m rows and m columns, then Arr.shape is (m,n). 
        # So Arr.shape[0] is m and Arr.shape[1] is n. Also, Arr.shape[-1] is n, Arr.shape[-2] is m.
        assert x.shape[-1] == self.ndim
        assert y.shape[-1] == self.ndim
        if pairwise:
            x, y = self._pairwise(x, y)
        #高斯径向基函数    
        d = self.params[1:] * (x - y) ** 2
        return self.params[0] * np.exp(-0.5 * np.sum(d, axis=-1))

    def derivatives(self, x, y, pairwise=True):
        if pairwise:
            x, y = self._pairwise(x, y)
        d = self.params[1:] * (x - y) ** 2
        delta = np.exp(-0.5 * np.sum(d, axis=-1)) #对c0求导
        deltas = -0.5 * (x - y) ** 2 * (delta * self.params[0])[:, :, None] #对c1求导
        return np.concatenate((np.expand_dims(delta, 0), deltas.T)) #[c0,c1]

    def update_parameters(self, updates):
        self.params += updates
