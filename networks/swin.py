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
from networks.swin_functions import window_partition, window_reverse

class PatchEmbedding(Layer):
    """ 
    Args:
        img_size (int): Image size, 224 is set by default.
        patch_size (tuple[int]): Size of patch, 16 is set by default.
        embed_dim (int): Dimension for embedding output
        name(str): Model Name.
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, name=None):
        super(PatchEmbedding, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 32 by default.

        self.proj = Conv2D(filters=embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='SAME',
                                  bias_initializer=initializers.Zeros())

    def call(self, inputs):
        batch_size, height, width, channel = inputs.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(inputs)
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, (-1, self.num_patches, self.embed_dim))
        return x

class MlpBlock(Layer):
    """
        FFN in paper Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
        
        Args:
            dropout_rate: Ratio of Dropout
            hidden_unit: out channel list
    """
    def __init__(self, hidden_unit: list, dropout_rate: float = 0., name: str = "mlp_block"):
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
                 qk_scale=None,
                 attn_drop: float = 0., 
                 proj_drop: float = 0., 
                 name = None):
        super(WindowAttention, self).__init__(name=name)
        self.dim = dim
        self.head_dim = dim // num_heads
        self.all_head_dims = self.head_dim * num_heads
        self.num_heads = num_heads
        self.scale = qk_scale if qk_scale != None else self.head_dim ** -0.5
        # self.scale = self.head_dim ** -0.5
        self.qkv = Dense(self.all_head_dims*3, name="attention_qkv")
        self.proj = Dense(dim, name="attention_proj")
        self.attention_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)
        self.windows_size = window_size

    def call(self, x: tf.Tensor, mask=None, return_attns=False, training=True):
        # x'shape -> [B, num_patches, embed_dim]
        B_, N, C = x.shape
        qkv = self.qkv(x)
        # qkv'shape -> [B, num_patches, all_head_dims * 3]
        qkv = tf.reshape(qkv, (-1, N, 3, self.num_heads, self.all_head_dims // self.num_heads))
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        # q/k/v 's shape -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        query, key, value = qkv[0], qkv[1], qkv[2]
        # attention' shape -> [batch_size, num_heads, num_patches+1, num_patches+1]
        attention = tf.matmul(a = query, b = key, transpose_b=True) * self.scale
        #TODO relative_position_bias.
        if mask is not None:
            attention = tf.nn.softmax(attention, axis=-1)
        else: 
            attention = tf.nn.softmax(attention, axis=-1)

        attention = self.attention_drop(attention, training=training)
        # x'shape -> [batch_size, num_heads, num_patches, embed_dim_per_head]
        x = tf.matmul(attention, value)
        # x'shape -> [batch_size, num_patches, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # x'shape -> [batch_size, num_patches, num_heads* embed_dim_per_head]
        x = tf.reshape(x, (-1, N, self.all_head_dims))
        # final projection
        
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attns:
            return x, attention
        else:
            return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "dim": self.dim,
            "windows_size": self.windows_size,
            "num_heads": self.num_heads,
            "all_head_dims":self.all_head_dims,
            "scale": self.scale
            # TODO win
            })

        return config


class SwinTransformerBlock(Layer):
    """
        Swin Transformer Block including window partition, mlp layers and window based attention layer.
        
        Args:
            input_resolution: Input resolution
            dim: Embedding dimension
            windows_size: Window size
            num_heads: Number of heads
            qkv_bias: If True, add a learnabel bias to query, key and value, True is set by default.
            attn_drop: Dropout ratio of attention weights. 0.0 is set by default.
            common_drop: Dropout ratio for projection and mlp layers. 0.0 is set by default.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            drop_path: Stochastic depth rate. 0.0 is set by default.
            shift_size: Shift size for SW-MSA
    """
    def __init__(self,
                 input_resolution: tuple,
                 dim: int, 
                 windows_size: int = 7,
                 num_heads: int = 8,
                 qkv_bias: bool =True,
                 qk_scale: float=None,
                 attn_drop: float= 0.0,
                 common_drop: float = 0.0,
                 mlp_ratio: float = 4.0,
                 drop_path: float = 0.0,
                 shift_size: int = 0, 
                 name = None):
        super(SwinTransformerBlock, self).__init__(name = name)
        self.dim = dim
        self.input_resolution = input_resolution
        self.windows_size = windows_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.windows_size:
            self.shift_size = 0
            self.windows_size = min(self.input_resolution)
        assert(0 <= self.shift_size < self.windows_size), "shift_size must in 0-window_size"
        self.norm1 = LayerNormalization(epsilon=1e-6, name='layer_norm_attn')
        self.attention_layer = WindowAttention(
            dim = dim,
            window_size=windows_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale= qk_scale,
            attn_drop=attn_drop,
            proj_drop=common_drop,
            name="window_attention"
                )
    
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else tf.identity
        self.norm2 = LayerNormalization(epsilon=1e-6, name='layer_norm_mlp')
        self.mlp_layers = MlpBlock(dropout_rate=common_drop, hidden_unit= [int(dim * mlp_ratio), dim])
        
        # TODO shift_windows
        if self.shift_size > 0:
            self.attn_mask = self.get_attn_mask()
        else:
            self.attn_mask = None
    def call(self,  x: tf.Tensor, return_attns=False):
        H, W =  self.input_resolution
        B, N, C = x.shape[0], x.shape[1], x.shape[2]
        
        shortcut = x
        x = self.norm1(x) # x'shape -> [B, N, C]
        x = tf.reshape(x, (-1, H, W, C))

        #TODO cyclic shift
        if self.shift_size > 0:
            shifted_x = x
        else:
            shifted_x = x

        # x_window'shape -> [num_win*B, window_size, window_size, C]
        x_windows = window_partition(shifted_x, self.windows_size)
        x_windows = tf.reshape(x_windows, (-1, self.windows_size * self.windows_size, C))
        
        # attn_windows'shape -> [num_win* B, window_size * window_size, C]
        if not return_attns:
            attn_windows = self.attention_layer(x_windows, mask=self.attn_mask)
        else:
            attn_windows, attn_scores = self.attention_layer(x_windows, mask=self.attn_mask, return_attns=True)
        #attn_windows'shape -> [num_win*B, window_size, window_size, C]
        attn_windows = tf.reshape(attn_windows, (-1, self.windows_size, self.windows_size, C))
        # shifted_x'shape -> [B, H, W, C]
        shifted_x = window_reverse(attn_windows, self.windows_size, H, W)
        #TODO reverse cyclic shift
        if self.shift_size > 0:
            x = shifted_x
        else:
            x = shifted_x
        # x'shape -> [B, H*W, C]
        x = tf.reshape(x, (-1, H*W, C))

        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        x = self.mlp_layers(x)
        x = self.drop_path(x)
        x = shortcut + x
        
        if return_attns:
            return x, attn_scores
        else:
            return x

    def get_attn_mask(self):
        return None



class PatchMerging(Layer):
    """
        Patch Merging Layer.

        Args:
             input_resolution (tuple[int]): Resolution of input feature.
             dim(int): Number of input channels
    """
    def __init__(self, input_resolutions: tuple, embed_dim: int, out_dim: int=None, name: str=None, **kwargs):
        super(PatchMerging, self).__init__(name=name)
        self.input_resolution = input_resolutions
        self.embed_dim = embed_dim
        self.out_dim = out_dim or 2 * embed_dim
        self.norm = LayerNormalization(epsilon=1e-6, name='layer_norm_patch_merging')
        self.reduction = Dense(self.out_dim, use_bias=False)
    
    def call(self, x: tf.Tensor):
            # x: B, H*W, C
            H, W = self.input_resolution
            B, _, C = x.shape[0], x.shape[1], x.shape[2] # _ stands for number of patches.
            x = tf.reshape(x, (-1, H, W, C))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = tf.concat([x0, x1, x2, x3], axis=-1) # x'shape -> [B, H/2, W/2, 4*C]
            x = tf.reshape(x, (-1, x.shape[1] * x.shape[2], 4 * C))
            x = self.norm(x)
            x = self.reduction(x) # x'shape -> [B, H*W/4, out_dim]
            return x

class StochasticDepth(Layer):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = float(drop_prob)
    
    def call(self, x, training = False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x/keep_prob) * random_tensor
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "drop_prob": self.drop_prob
            })
        return config
        
