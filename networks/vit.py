from ast import Mult
import imp
# from keras_layer_normalization import LayerNormalization
import tensorflow as tf
from networks.vit_functions import gelu
import numpy as np

class PatchEmbedding(tf.keras.layers.Layer):

    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, name=None):
        super(PatchEmbedding, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.img_size = (img_size, img_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 32 by default.

        self.proj = tf.keras.layers.Conv2D(filters=embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='SAME',
                                  bias_initializer=tf.keras.initializers.Zeros())

    def call(self, inputs):
        batch_size, height, width, channel = inputs.shape
        assert height == self.img_size[0] and width == self.img_size[1], \
            f"Input image size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(inputs)
        # [B, H, W, C] -> [B, H*W, C]
        x = tf.reshape(x, (-1, self.num_patches, self.embed_dim))
        return x

class MultHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 dim, 
                 num_heads=8, 
                 qkv_bias=False, 
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0., 
                 name=None):
        # self.kernel_init_strategy = tf_keras.initializers.GlorotUniform()
        # self.bias_init_strategy = tf_keras.initializers.Zeros()
        self.kernel_init_strategy = 'glorot_uniform'
        self.bias_init_strategy = tf.keras.initializers.Zeros()
        
        super(MultHeadAttentionLayer, self).__init__(name=name)
        self.num_heads = num_heads
        head_dim = int(dim // self.num_heads)
        self.all_head_dims = head_dim * self.num_heads
        self.scale = qk_scale if qk_scale != None else head_dim ** -0.5
        qkv_dim = self.all_head_dims*3
        self.qkv = tf.keras.layers.Dense(qkv_dim, use_bias=qkv_bias, name="qkv", 
                         kernel_initializer=self.kernel_init_strategy, 
                         bias_initializer=self.bias_init_strategy)
        self.attention_drop = tf.keras.layers.Dropout(attn_drop_ratio)
        self.proj = tf.keras.layers.Dense(dim, 
                          name="out",
                         kernel_initializer=self.kernel_init_strategy, 
                         bias_initializer=self.bias_init_strategy)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop_ratio)

    def call(self, inputs, training=None):
        # input' shape -> [batch_size, num_patches + 1, embed_dim]
        B, N, _ = inputs.shape 
        # qkv'shape -> [batch_size, num_patches + 1, 3 * all_head_dim]
        qkv = self.qkv(inputs)
        # split q, k, v and its corresponding head.
        qkv = tf.reshape(qkv, [-1, N, 3, self.num_heads, self.all_head_dims // self.num_heads])
        # qkv'shape is transposed to [3, batch_size, num_heads, num_patches+1, embed_dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        # q/k/v 's shape -> [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        # transposed key's shape -> [batch_size, num_heads, embed_dim_per_head, num_patches+1]
        query, key, value = qkv[0], qkv[1], qkv[2]
        # attention' shape -> [batch_size, num_heads, num_patches+1, num_patches+1]
        attention = tf.matmul(a = query, b = key, transpose_b=True) * self.scale
        attention = tf.nn.softmax(attention, axis=-1)
        attention = self.attention_drop(attention, training=training)
        # x'shape -> [batch_size, num_heads, num_patches+1, embed_dim_per_head]
        x = tf.matmul(attention, value)
        # x'shape -> [batch_size, num_patches+1, num_heads, embed_dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [-1, N, self.all_head_dims])
        x = self.proj(x)
        # projected x'shape -> [B, N, self.dim] by weights[B, N, self.all_head_dims, dim]
        x = self.proj_drop(x, training=training)
        return x

class MLP(tf.keras.layers.Layer):

    def __init__(self, in_features, mlp_ratio=4.0, drop=0., name=None):
        self.kernel_init_strategy = 'glorot_uniform'
        self.bias_init_strategy = tf.keras.initializers.RandomNormal(stddev=1e-6)
        super(MLP, self).__init__(name=name)
        self.fc1    = tf.keras.layers.Dense(int(in_features * mlp_ratio), name="dense_0",
                             kernel_initializer=self.kernel_init_strategy, 
                             bias_initializer=self.bias_init_strategy)
        self.act    = gelu
        self.fc2    = tf.keras.layers.Dense(in_features, name="dense_1",
                         kernel_initializer=self.kernel_init_strategy, 
                         bias_initializer=self.bias_init_strategy)
        self.drop   = tf.keras.layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

class ClassToken_PosEmbed(tf.keras.layers.Layer):
    def __init__(self, embed_dim= 768, num_patches=196, name=None):
        super(ClassToken_PosEmbed, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def build(self, input_shape):
        self.cls_token = self.add_weight(name='cls_token',
            shape=[int(1), int(1), int(self.embed_dim)], 
            initializer=tf.keras.initializers.Zeros(), 
            trainable=True,
            dtype=tf.float32)
        # self.cls_token = tf.Variable(name='cls_token', 
        #                             initial_value=tf.zeros_initializer(shape=(1, 1, self.embed_dim)))
        self.pos_embed = self.add_weight(name='pos_embed',
            shape=[1, self.num_patches+1, self.embed_dim], 
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02), 
            trainable=True,
            dtype=tf.float32)

    def call(self, inputs):
        batch_size, _, _ = inputs.shape
        print("batch_size -> ", batch_size)
        # TODO fix this bug.
        cls_token = tf.broadcast_to(self.cls_token, shape=[batch_size, int(1), int(self.embed_dim)])
  
        x = tf.concat([cls_token, inputs], axis=1)
        x = x + self.pos_embed
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, 
                 dim, 
                 num_heads=8,
                 qkv_bias=False, 
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0., 
                 name=None):
        super(Encoder, self).__init__(name=name)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_0')
        self.attention = MultHeadAttentionLayer(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                                           attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                                           name='mult_head_self_attention')
        self.drop_path = tf.keras.layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1)) if drop_path_ratio > 0. \
            else tf.keras.layers.Activation('linear')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')
        self.mlp = MLP(dim, drop=drop_ratio, name='mpl_block')
    
    def call(self, inputs, training=None):
        x = self.norm1(inputs)
        x = self.attention(x)
        if isinstance(self.drop_path, tf.keras.layers.Activation):
            x = self.drop_path(x)
        else:
            x = self.drop_path(x, training=training)
        x = inputs + x
        x1 = self.norm2(x)
        x1 = self.mlp(x1) #! change x to x1
        if isinstance(self.drop_path, tf.keras.layers.Activation):
            x1 = self.drop_path(x1)
        else:
            x1 = self.drop_path(x1, training=training)
        x1 = x + x1
        return x1
    
class VisionTransformer(tf.keras.Model):
    def __init__(self, img_size = 224, patch_size = 16, embed_dim = 768, 
                 depth=12, num_heads=8, qkv_bias=True, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                 representation_size=None, num_classes=10, name="ViT-B/16"):
        super(VisionTransformer, self).__init__(name=name)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.qkv_bias = qkv_bias
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, name='patch_embed')
        num_patches = self.patch_embed.num_patches
        self.cls_token_pos_embed = ClassToken_PosEmbed(embed_dim=embed_dim, 
                                                       num_patches=num_patches, 
                                                       name='cls_pos')
        self.pos_drop = tf.keras.layers.Dropout(drop_ratio)
        dpr = np.linspace(0., drop_path_ratio, depth)
        self.blocks = [Encoder(dim=embed_dim, num_heads=num_heads, 
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, 
                               drop_path_ratio=0., name="encoder_block_{}".format(i)) for i in range(depth)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")
        if representation_size:
            self.has_logits = True
            self.pre_logits = tf.keras.layers.Dense(representation_size, activation="tanh", name="pre_logits")
        else:
            self.has_logits = False
            self.pre_logits = tf.keras.layers.Activation("linear")
        self.head = tf.keras.layers.Dense(num_classes, name="head", kernel_initializer=tf.keras.initializers.Zeros())
    
    def call(self, inputs, training=None):
        # input'shape -> [B, H, W, C]
        x = self.patch_embed(inputs) # [B, num_patches, embed_dim]
        x = self.cls_token_pos_embed(x) # [B, num_patches+1, embed_dim]
        x = self.pos_drop(x, training=training)
        #! Encoder Problem
        for i, block in enumerate(self.blocks):
            x = block(x, training=training)
        #! LayerNormalization
        x = self.norm(x)
        x = self.pre_logits(x[:, 0]) # [B, 1, embed_dim] or [B, 1, representation_size]
        x = self.head(x) # [B, 1, num_classes]
        return x
        


    
           