import numpy as np
import cv2
import sys
sys.path.append('..')
# from networks.vit import MLP, Encoder, PatchEmbedding, MultHeadAttentionLayer, \
#                         ClassToken_PosEmbed, Encoder, VisionTransformer
# directory of cifar dataset.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import backend as K
# from tensorflow.keras import Input, Model
from networks.swin import MlpBlock, WindowAttention, SwinTransformerBlock, PatchMerging, StochasticDepth, PatchEmbedding
from networks.swin_functions import window_partition, window_reverse
test_data_dir =  "/Users/crystal-dragon-lyb/CrystalNetDataset/cifar/test/"

def map_log_color(color):
    foreground_color= {
        'cyan': '36',
        'red': '31',
        'pale yellow': '33', 
        'blue': '34',
        'deep yellow': '32',
        'lilac': '35',
    }
    return foreground_color[color]

def log_info(content):
    color = map_log_color('blue')
    print("\033[1;"+color+"m{}\033[0m".format(content))

def general_test(function, test_name: str):
    log_info('-- running test <' + test_name + '>')
    out = function()
    print("-- out'shape -> ", str(out.shape))
    log_info('-- test <' + test_name + '> finished')

def patch_embedding_test():
    test_data = tf.ones(shape=(1, 224, 224, 3))
    patch_embed = PatchEmbedding(patch_size = 16, embed_dim=768)
    out = patch_embed(test_data)
    return out

def attention_test():
    test_data = tf.ones(shape=(4, 49, 768))
    attn_model = WindowAttention(dim=768, 
            window_size=7,
            num_heads=6)
    out = attn_model(test_data)
    return out

def window_partition_test():
    test_data = tf.ones(shape=(1, 14, 14, 768))
    x_windows = window_partition(test_data, 7)
    return x_windows

def mlp_test():
    test_data = tf.ones(shape=(1, 196, 768))
    mlp_model = MlpBlock([int(768*4.0), 768])
    out = mlp_model(test_data)
    return out

def encoder():
    test_data = tf.ones(shape=(4, 3136, 96))
    swin_transformer_block = SwinTransformerBlock(
            input_resolution=(56, 56),
            dim=96,
            windows_size=7,
            num_heads=6
            )
    out = swin_transformer_block(test_data)
    return out

def vit_test():
    # data = cv2.imread(test_data_dir + '0_cat.png')
    # data = (data / 255. - 0.5) / 0.5
    # data = np.expand_dims(data, 0)
    # print(data.shape)
    # vit_model = VisionTransformer(img_size=32, patch_size=8, embed_dim=96, depth=12, num_heads=6)
    # vit_model.build(input_shape=(1, 32, 32, 3))
    # out = vit_model.predict(data, batch_size = 1)
    # return out
    pass

def main():
    general_test(patch_embedding_test, 'patch embedding')
    general_test(mlp_test, 'mlp layer')
    general_test(window_partition_test, 'window partition')
    general_test(attention_test, 'attention')
    general_test(encoder, 'encoder block')
    #general_test(vit_test, "vision transformer")
    
if __name__ == "__main__":
    main()

