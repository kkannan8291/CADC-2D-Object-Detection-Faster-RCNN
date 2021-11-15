import os
import random

import tensorflow.compat.v2 as tf
import numpy as np
from inference.common import measure

path_to_npz ='/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/models/moco-preTrainedModel/MoCo_v2.npz'


sess = tf.compat.v1.InteractiveSession()
measure(lambda: tl.files.load_and_assign_npz_dict(path_to_npz, sess), 'load npz')
frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
tf_model = tf.graph_util.remove_training_nodes(frozen_graph)
uff_model = measure(lambda: uff.from_tensorflow(tf_model, output_names), 'uff.from_tensorflow')
print('uff model created')