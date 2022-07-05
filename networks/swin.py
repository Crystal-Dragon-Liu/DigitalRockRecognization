# --------------------------------------------------------
# Swin Transformer
# Licensed under The MIT License [see LICENSE for details]
# Written by Yubo Liu
# --------------------------------------------------------

import enum
from re import L
import tensorflow as tf
from networks.vit_functions import gelu
import numpy as np
from tensorflow.keras.layers import Layer, Dense, Dropout, Conv2D, LayerNormalization
from tensorflow.keras import Sequential
from tensorflow.keras import initializers

class MlpBlock(Layer):
    """
        FFN in paper Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    """
    def __init__(self, dropout_rate: float, hidden_unit: list, name: str = "mlp_block"):
         super(MlpBlock, self).__init__(name=name)
         self.ffn = [Dense(units, activation=gelu if idx ==0 else None, bias_initializer=initializers.RandomNormal(stddev=1e-6))  \
            for (idx, units) in enumerate(hidden_unit)]
         self.dropout_list = [Dropout(dropout_rate) for i in range(len(hidden_unit))]
    def call(self, inputs):
        x = inputs
        for id, ffn_layer in enumerate(self.ffn):
            x = ffn_layer(x)
            x = self.dropout_list[id](x)
        return x

class WindowAttention(Layer):
    """
        Window based multi-head self attention (W-MSA) module with relative position bias.

        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            windows_size: the height and width of the window
            qkv_bias: If True, add a learnable bias to query, key and value. True is set by default.
            attn_drop: Dropout ratio of attention weight. 0. is set by default.
            proj_drop: Dropout ratio of output. 0. is set by default.
    """
    def __init__(self, dim: int, window_size: tuple,
                 num_heads: int = 8, qkv_bias: bool = True, 
                 attn_drop: float = 0., 
                 proj_drop: float = 0.):
                 pass

class PatchMerging(Layer):
    """
        Patch Merging Layer.

        Args:
             input_resolution (tuple[int]): Resolution of input feature.
             dim(int): Number of input channels
    """
    def __init__(self, input_resolutions: tuple(int), embed_dim: int, out_dim: int=None, name: str=None, **kwargs):
        super(PatchMerging, self).__init__(name=name)
        self.input_resolution = input_resolutions
        self.embed_dim = embed_dim
        self.out_dim = out_dim or 2 * embed_dim
        self.norm = LayerNormalization(epsilon=1e-6, name='layer_norm_patch_merging')
        self.reduction = Dense(self.out_dim, use_bias=False)
    
    def call(self, x):
            # x: B, H*W, C
            H, W = self.input_resolution
            B, _, C = x.shape[0], x.shape[1], x.shape[2] # _ stands for number of patches.
            x = tf.reshape(x, (B, H, W, C))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = tf.concat([x0, x1, x2, x3], axis=-1) # x'shape -> [B, H/2, W/2, 4*C]
            x = tf.reshape(x, (B, -1, 4 * C))
            x = self.norm(x)
            x = self.reduction(x) # x'shape -> [B, H*W/4, out_dim]
            return x


