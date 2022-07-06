import numpy as np
import cv2
import sys
sys.path.append('..')
from networks.vit import MLP, Encoder, PatchEmbedding, MultHeadAttentionLayer, \
                        ClassToken_PosEmbed, Encoder, VisionTransformer
# directory of cifar dataset.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import backend as K
# from tensorflow.keras import Input, Model

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
    patch_embed = PatchEmbedding(patch_size=16, embed_dim=768)
    out = patch_embed(test_data)
    return out

def attention_test():
    test_data = tf.ones(shape=(1, 16, 96))
    attn_model = MultHeadAttentionLayer(96, 6)
    out = attn_model(test_data)
    return out

def class_token_and_pos_embed_test():
    test_data = tf.ones(shape=(1, 16, 96))
    cls_pos_embed = ClassToken_PosEmbed(embed_dim=96, num_patches=16)
    out = cls_pos_embed(test_data)
    return out

def mlp_test():
    test_data = tf.ones(shape=(1, 17, 96))
    mlp = MLP(in_features=96)
    out = mlp(test_data)
    return out

def encoder():
    test_data = tf.ones(shape=(1, 17, 96))
    encoder = Encoder(96, 6)
    out = encoder(test_data)
    return out

def vit_test():
    data = cv2.imread(test_data_dir + '0_cat.png')
    data = (data / 255. - 0.5) / 0.5
    data = np.expand_dims(data, 0)
    print(data.shape)
    vit_model = VisionTransformer(img_size=32, patch_size=8, embed_dim=96, depth=12, num_heads=6)
    vit_model.build(input_shape=(1, 32, 32, 3))
    out = vit_model.predict(data, batch_size = 1)
    return out

def main():
    general_test(patch_embedding_test, 'patch embedding')
    general_test(attention_test, 'attention')
    general_test(class_token_and_pos_embed_test, 'class_token and position_embed')
    general_test(mlp_test, 'MLP')
    general_test(encoder, 'encoder block')
    general_test(vit_test, "vision transformer")
    
if __name__ == "__main__":
    main()

