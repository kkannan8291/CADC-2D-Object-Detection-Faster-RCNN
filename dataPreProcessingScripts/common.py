
import numpy as np
import tensorflow.compat.v2 as tf
import cv2

regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

def measure(f, name=None):
    if not name:
        name = f.__name__
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    _default_profiler(name, duration)
    return result
