import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.specs.python.summaries import tf_left_split

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def input_fn():
    
    
    return (features, labels)



def model_fn(features, labels, mode, params, config):
    
    
    return tf.estimator.EstimatorSpec(predictions = predictions, mode = mode, loss = loss, train_op = train_op, scaffold = scaffold)





estimator = tf.estimator.Estimator(model_fn, model_dir = 'test')



estimator.train(input_fn, steps = 10)








