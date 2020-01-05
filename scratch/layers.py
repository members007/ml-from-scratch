from __future__ import print_function, division
import math
import numpy as np
import copy
from mlfromscratch.deep_learning.activation_functions import Sigmoid, ReLU, SoftPlus, LeakyReLU
from mlfromscratch.deep_learning.activation_functions import TanH, ELU, SELU, Softmax

class Layer(object):
    def set_input_shape(self,shape):
        self.input_shape=shape
    
    def layer_name (self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError

    def output_shape(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units=n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self,optmizer):
        limit = 1/ math.sqrt(self.input.shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1,self.n_units))

        self.W_opt= copy.copy(optimizer)
        self.w0_opt=copy.copy(optimizer)

    def parameters(self):
        return np.prob(self.w.shape)+np.prob(self.w0.shape)

    def forward_pass(self,X,training = True):
        self.layer_input=X
        return X.dot(self.X)+self.w0
    
    def backward_pass(self, accum_grad):
        W=self.W

        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad,axis=0,keepdims=True)

            self.W=self.W_opt.update(self.W,grad_w)
            self.w0 = self.w0_opt.update(self.w0,grad_w0)

        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units, )
