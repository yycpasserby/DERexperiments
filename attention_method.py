# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:46:57 2019

@author: Administrator
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import keras   
import numpy as np
from itertools import combinations
import keras.backend as K   
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import RNN
from keras.layers import InputSpec
from keras.utils import conv_utils
from keras import activations

from collections import namedtuple

from data_config_preprocess import Config

    
class Attention(keras.layers.Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        print("attention intput :::_____________" , input_shape)

        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(self.attention_size,),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
#        
        et = K.tanh(K.dot(x, self.W) + self.b)
#        et = K.tanh(K.batch_dot(self.W, x) + self.b)
#        print(et.get_shape(),'et')
#        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
#        print(at.get_shape(),'at')
#        at = K.softmax(et, axis=1)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
#        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
#        print(atx.get_shape(),'atx')
        ot = atx * x
    
#        ot = K.batch_dot(at, x)
#        # output: (BATCH_SIZE, EMBED_SIZE)
#        output = K.sum(ot, axis=1)
#        print(ot.get_shape(),'ot')
        return ot

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])



class HLU(keras.layers.Layer):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x / (1 - x ) for x < 0`,
    `f(x) = x for x >= 0`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha: float >= 0. Negative slope coefficient.

    """

    def __init__(self, alpha=0.1, **kwargs):
        super(HLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)


    def call(self, inputs):

        inputs_m = tf.where(inputs < 0, inputs, K.zeros_like(inputs))
        
        return tf.where(inputs >= 0.0, inputs, inputs_m * self.alpha/(1 - inputs_m))

    def get_config(self):
        base_config = super(HLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Top_K(keras.layers.Layer):
    def __init__(self, k, sortable=True, **kwargs):
        self.K = k;
        self.sortable = sortable
        super(Top_K, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(Top_K, self).build(input_shape)
        
    def call(self, x, mask=None):
        x = tf.transpose(x, [0, 2, 1])
        x = tf.nn.top_k(x, k=self.K, sorted=self.sortable)[0]
        x = tf.transpose(x, [0, 2, 1])
        return x

    def compute_mask(self, inputs, input_mask=None):
        if input_mask is None:
            return input_mask
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.K, input_shape[-1])


class SelfAtt(keras.layers.Layer):
    def __init__(self, hiddensize=128,
                 use_weighted = False,
                 return_probabilities=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        self.attention_size = hiddensize
        self.use_weighted = use_weighted
        self.return_probabilities = return_probabilities
        self.kernel_regularizer = keras.layers.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.layers.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.layers.regularizers.get(activity_regularizer)
        super(SelfAtt, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.dims_int = int(input_shape[1]) * int(input_shape[2])
        self.dims_float = float(self.dims_int)
        self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                 shape=(int(input_shape[2]), self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.W2 = self.add_weight(name="W2_{:s}".format(self.name),
                                 shape=(self.attention_size, int(input_shape[1])),
                                 initializer="glorot_normal",
                                 trainable=True)
        if self.use_weighted is True:
            self.W3 = self.add_weight(name="W2_{:s}".format(self.name),
                                     shape=(int(input_shape[1]),1),
                                     initializer="glorot_normal",
                                     trainable=True)
        super(SelfAtt, self).build(input_shape)
        
    def call(self, x, mask=None):
        shape = x.get_shape().as_list()
        if len(shape) == 4:
            x = K.squeeze(x, axis=-1)
            
        OA = K.dot(x, self.W1)
        OA = K.tanh(OA)

        OA = K.dot(OA, self.W2)
        
        if mask is not None:
            mask = K.permute_dimensions(K.repeat(mask, OA.shape[-1]), [0,2,1]) #shape (none, repeat, x)
            OA *= K.cast(mask, K.floatx())
            
        OA = K.softmax(OA, axis=-2)    
        ot = K.batch_dot(K.permute_dimensions(x, [0,2,1]), OA)
        ot = K.permute_dimensions(ot, [0,2,1])
       
        if self.use_weighted is True:
            w = K.squeeze(K.repeat(self.W3, shape[-1]), axis=-1)

            ot = ot * w
        if self.return_probabilities:
            return OA
        else:
            return ot

    def get_config(self):
        config = {
            'kernel_regularizer': keras.layers.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.layers.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':keras.layers.regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(SelfAtt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_mask(self, input, input_mask=None):
        return input_mask

    def compute_output_shape(self, input_shape):
        return input_shape


class attention(keras.layers.Layer):
    def __init__(self, hiddensize = 128,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        self.attention_size = hiddensize
        # self.kernel_regularizer = keras.layers.regularizers.get(kernel_regularizer)
        # self.bias_regularizer = keras.layers.regularizers.get(bias_regularizer)
        # self.activity_regularizer = keras.layers.regularizers.get(activity_regularizer)
        super(attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.dims_int = int(input_shape[1])
        self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                 shape=(int(input_shape[1]), self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)

        super(attention, self).build(input_shape)
        
    def call(self, x, mask=None):
        shape = x.get_shape().as_list()
        if len(shape) == 4:
            x = K.squeeze(x, axis=-1)
        M = K.tanh(x)
        M = K.permute_dimensions(M, [0,2,1])
        A = K.dot(M, self.W1)
        # A = K.permute_dimensions(A, [0,2,1])
        # print(A.get_shape())

        OA = K.softmax(A, axis=-2)
        if mask is not None:
            mask = K.permute_dimensions(K.repeat(mask, OA.shape[-1]), [0,2,1]) #shape (none, repeat, x)
            OA *= K.cast(mask, K.floatx())
            
        ot = K.batch_dot(x, OA)

        return ot
    
    def compute_mask(self, input, input_mask=None):
        return input_mask

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], self.attention_size]
   

class Scaled_Dot_Product_Att(keras.layers.Layer):
    def __init__(self, hiddensize=128, return_probabilities=False, **kwargs):
        self.attention_size = hiddensize
        self.return_probabilities = return_probabilities
        super(Scaled_Dot_Product_Att, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.dims_int = int(input_shape[1]) * int(input_shape[2])
        self.dims_float = float(self.attention_size)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(3, int(input_shape[2]), self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Scaled_Dot_Product_Att, self).build(input_shape)
        
    def call(self, x, mask=None):
        shape = x.get_shape().as_list()
        if len(shape) == 4:
            x = K.squeeze(x, axis=-1)
        
        
        WQ = K.dot(x, self.W[0])
        WK = K.dot(x, self.W[1])
        WV = K.dot(x, self.W[2])

        at = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1])) / tf.sqrt(self.dims_float)
        

        if mask is not None and self.return_probabilities is False:
            mask = K.repeat(mask, at.shape[-2])#shape (none, repeat, x)
            at *= K.cast(mask, K.floatx())
            
        at = K.softmax(at, axis=-1)
        ot = K.batch_dot(at, WV)
        if self.return_probabilities:
            return at
        else:
            return ot

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.attention_size)
    
class MConcat_Reg(keras.layers.Layer):
    def __init__(self,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        self.kernel_regularizer = keras.layers.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.layers.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.layers.regularizers.get(activity_regularizer)
        super(MConcat_Reg, self).__init__(**kwargs)
    def build(self, input_shape):
        super(MConcat_Reg, self).build(input_shape)
   
    def call(self, x, mask=None):
        if len(x) > 1:
            new_x = K.concatenate(x, axis=-1)
        else:
            new_x = x[0]
        return new_x
 
    def get_config(self):
        config = {
            'kernel_regularizer': keras.layers.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.layers.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':keras.layers.regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(MConcat_Reg, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            print('A `Concatenate` layer should be called '
                             'on a list of inputs.')
            return input_shape
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[-1] is None or shape[-1] is None:
                output_shape[-1] = None
                break
            output_shape[-1] += shape[-1]
        return tuple(output_shape)


class Reg1(keras.regularizers.Regularizer):

    def __init__(self, alpha=0., sizes=[1,2,3]):
        self.alpha = K.cast_to_floatx(alpha)
        self.sizes = sizes

    def __call__(self, x):

        shap = x.get_shape().as_list()
        regularization = 0.
        if self.alpha:
            xt = K.permute_dimensions(x,[0,2,1])
            xp = K.softmax(K.batch_dot(xt,x))
            size = xp.get_shape().as_list()
            fr = K.sum(K.square(xp - tf.eye(num_rows=size[1], num_columns=size[2], batch_shape=size[0], dtype=xp.dtype)), axis=[-2,-1])
            regularization += K.mean(fr / size[-1])
        
            regularization = self.alpha * regularization
        return  regularization

    def get_config(self):
        return {'alpha': float(self.alpha)}


class Reg(keras.regularizers.Regularizer):

    def __init__(self, alpha=0., sizes=[1,2,3]):
        self.alpha = K.cast_to_floatx(alpha)
        self.sizes = sizes

    def __call__(self, x):

        x = tf.split(axis=-1, value=x, num_or_size_splits=self.sizes) #value, num_or_size_splits, axis=0,
        regularization = 0.
        if self.alpha:
            k = 0.0
            for i in range(len(self.sizes)):
                xt = K.permute_dimensions(x[i],[0,2,1])
                for j in range(len(self.sizes)):
                    if i == j:
                        continue
                    k+=1.0
                    xp = K.softmax(K.batch_dot(xt,x[j]))
                    size = xp.get_shape().as_list()
                    fr = K.abs(K.square(xp - tf.eye(num_rows=size[1], num_columns=size[2], batch_shape=size[0], dtype=xp.dtype))) / K.sqrt(size[1] * size[2])
                    r = K.mean(K.sum(self.alpha * fr, axis=[-2,-1]))
                    regularization += r
        
            regularization = self.alpha * regularization / k
        return  regularization

    def get_config(self):
        return {'alpha': float(self.alpha)}

class FREG(keras.regularizers.Regularizer):

    def __init__(self, alpha=0.):
        self.alpha = K.cast_to_floatx(alpha)

    def __call__(self, x):
        regularization = 0.
        if self.alpha:

            xt = K.permute_dimensions(x,[0,2,1])
            xp = K.batch_dot(x, xt)
            size = xp.get_shape().as_list()
            fr = K.abs(K.square(xp - tf.eye(num_rows=size[1], num_columns=size[2], batch_shape=size[0], dtype=xp.dtype)))
            r = K.sum(self.alpha * fr)
            regularization += r

        return regularization



class CreateMask(keras.layers.Layer):
    """Create mask from input tensor.
    The shape of the mask equals to the shape of the input tensor.
    # Input shape
        Tensor with shape: `(batch_size, ...)`.
    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, mask_value=0., **kwargs):
        super(CreateMask, self).__init__(**kwargs)
        self.supports_masking = True
        self.mask_value = mask_value

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, self.mask_value)

    def call(self, inputs, **kwargs):
        return K.zeros_like(inputs)

    def get_config(self):
        base_config = super(CreateMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RemoveMask(keras.layers.Layer):
    """Remove mask from input tensor.
    # Input shape
        Tensor with shape: `(batch_size, ...)`.
    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, **kwargs):
        super(RemoveMask, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, **kwargs):
        return K.identity(inputs)


class RestoreMask(keras.layers.Layer):
    """Restore mask from the second tensor.
    # Input shape
        Tensor with shape: `(batch_size, ...)`.
        Tensor with mask and shape: `(batch_size, ...)`.
    # Output shape
        Tensor with shape: `(batch_size, ...)`.
    """

    def __init__(self, used=True, **kwargs):
        super(RestoreMask, self).__init__(**kwargs)
        self.used = used
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def compute_mask(self, inputs, mask=None):
        return mask[1]

    def call(self, inputs, mask=None, **kwargs):
        out = inputs[0]
        if self.used is True:
            if len(out.get_shape().as_list()) > 3:
                out = K.squeeze(out, axis=-1)

            mask = mask[1]
            mask = K.permute_dimensions(K.repeat(mask, out.shape[-1]), [0,2,1]) #shape (none, repeat, x)
            out *= K.cast(mask, K.floatx())
        return K.identity(out)

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


        