import tensorflow as tf, numpy as np

def test_type(l):
    if isinstance(l,list):
        print("It is a list with length:", len(l))
    elif isinstance(l,np.ndarray):
        print("It is an array with shape:", l.shape)
    elif tf.is_tensor(l):
        print("It is a tensor of shape:", l.get_shape())
    else:
        raise Exception('wrong type')