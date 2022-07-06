import tensorflow as tf

def window_partition(x: tf.Tensor, windows_size: int):
    """
        Args:
            x: tensor with shape [B, H, W, C]
            windows_size: window size
        Return:
            windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // windows_size, windows_size, W // windows_size, windows_size, C])
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, [-1, windows_size, windows_size, C])
    return windows

def window_reverse(windows: tf.Tensor, windows_size: int, H: int, W: int):
    """
        Args:
            windows: Tensor with size (num_windows*B, windows_size, windows_size, C)
            windows_size: windows size
            H: height of image
            W: width of image
        Returns:
            x: Tensor with shape (B, H, W, C)
    """
    B = int(windows.shape[0] // (H / windows_size * W / windows_size))
    x = tf.reshape(windows, (B, H//windows_size, W // windows_size, windows_size, windows_size, -1))
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (B, H, W, -1))


