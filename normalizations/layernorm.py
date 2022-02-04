import numpy as np
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Performs Layer Normalization"""
    def __init__(self,center, scale, **kwargs):
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        '''creates the layer's variables and is called the first time the layer is used'''
        self.beta = self.add_weight(name='beta', shape=[input_shape[-1:]], dtype=tf.float32, initializer="zeros")
        self.gamma = self.add_weight(name='gamma', shape=[input_shape[-1:]], dtype=tf.float32, initializer="ones")
        super().build(input_shape)

    def call(self, X):
        e = 0.001
        mean, var = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.gamma @ (X - mean) / (tf.math.sqrt(var) + e) + self.beta
