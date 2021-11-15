import os
import random

import tensorflow as tf
import numpy as np

path_to_npz ='/home/kaushik/ObjectDetection/CADC-2D-Object-Detection-Faster-RCNN/models/moco-preTrainedModel/MoCo_v2.npz'


def convert_param_name(param):
    # Get the variable name in the .npz file
    # Compare with the variable name of the network
    # Get the network variable name: a dictionary such as the variable value in the file
    # print('--> convert_param_name ...')
    resnet_param = {}
    for k in param.keys():
        # print(k) 
        var_name = k.replace('W','weights')
        var_name = var_name.replace('bn','BatchNorm')
        resnet_param[var_name] = param[k]
    return resnet_param


def initial_imagenet(sess, path_to_npz):
    print('Initializing through npz file trained on ImageNet ...')
    sess.run(tf.global_variables_initializer()) # Initialize the network first to avoid some network variables that do not exist in the file
    param = np.load(path_to_npz, encoding='latin1') # load file
    param = convert_param_name(param) # Get the network variable name: a dictionary like the variable value in the file
    for var in tf.trainable_variables(): # Assign values ​​to network variables
        if var.name in param.keys():
            sess.run(var.assign(param[var.name]))
            # print(var.name, var.shape, param[var.name].shape)
            # print(var.name,'done')
        else:
            print(var.name, var.shape,'not in trained weights ---------------------------------')
            pass

def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #random.seed(args.seed)

    #model = xxModel(num_classes=2)

    with tf.Session() as sess:
        initial_imagenet(sess=sess, path_to_npz=args.npz_file)
        #train(sess, model, train_set, val_set, args.checkpoint, **train_kwargs(args))
        ...


if __name__ =='__main__':
    main()