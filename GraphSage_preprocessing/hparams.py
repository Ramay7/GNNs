# -*- coding: utf-8 -*-

import tensorflow as tf
import time, os, sys

flags = tf.app.flags
FLAGS = flags.FLAGS

model_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

flags.DEFINE_string('model_date', model_date, 'date for training model')
flags.DEFINE_string('save_name', './saved_model/' + model_date + '/', 'Path for saving model')
flags.DEFINE_string('dataset', 'toy-ppi', 'Dataset string.')
flags.DEFINE_string('data_prefix', './data', 'Datapath prefix.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2048, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('num_clusters', 50, 'Number of clusters.')
flags.DEFINE_integer('bsize', 1, 'Number of clusters for each batch.')
flags.DEFINE_integer('num_clusters_val', 5, 'Number of clusters for validation.')
flags.DEFINE_integer('num_clusters_test', 1, 'Number of clusters for test.')
flags.DEFINE_integer('num_layers', 5, 'Number of GCN layers.')
flags.DEFINE_float('diag_lambda', 1, 'A positive number for diagonal enhancement, -1 indicates normalization without diagonal enhancement')
flags.DEFINE_bool('multilabel', True, 'Multilabel or multiclass.')
flags.DEFINE_bool('layernorm', True, 'Whether to use layer normalization.')
flags.DEFINE_bool('precalc', True, 'Whether to pre-calculate the first layer (AX preprocessing).')
flags.DEFINE_bool('validation', True, 'Print validation accuracy after each epoch.')

def create():
    return FLAGS