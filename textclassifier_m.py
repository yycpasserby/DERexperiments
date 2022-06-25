# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import keras
import sklearn.metrics as sm
import numpy as np
from operator import truediv
from data_config_preprocess import Config
import attention_method as am

      
class ConV(object):
    
#
#    def __init__(self, config, word_embedding):
##        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
#        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
#        self.dropout_rate = config.model.dropout_rate
#
#        with tf.name_scope('embbeding'):
#            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
#                                                            input_length=config.word.sentence_len, weights=[word_embedding],
#                                                            mask_zero=False, trainable=False, name='word_embbeding')
#
#            self.embedded_words = self.W(self.input_X)
#        
##        shap = self.embedded_words.get_shape().as_list()
##        output_ = keras.layers.Reshape([-1, shap[-1], 1])(self.embedded_words)
#        
#        output_ = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='relu')(self.embedded_words)
#        shap = output_.get_shape().as_list()
#        print(shap)
##        output_ = keras.layers.Reshape([-1, shap[-1] * shap[-2]])(output_)
#        
#        output_ = keras.layers.MaxPool1D(pool_size=shap[-2], strides=1)(output_)
#        
#        output_ = keras.layers.Flatten()(output_)
#        with tf.name_scope('output'):
#            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)
#
#
#        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)
#        self.model.summary()


    def __init__(self, config, word_embedding):   
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate
        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
            if config.model.spa_drop_rate != 0.0:
                self.embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(self.embedded_words)

        output_1 = keras.layers.Conv1D(filters=300, kernel_size=2, strides=1, activation='relu', padding='same')(self.embedded_words)
#        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
        
        output_2 = keras.layers.Conv1D(filters=300, kernel_size=3, strides=1, activation='relu', padding='same')(self.embedded_words)
#        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
        
        output_3 = keras.layers.Conv1D(filters=300, kernel_size=4, strides=1, activation='relu', padding='same')(self.embedded_words)
#        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
        
        output_ = keras.layers.Add()([output_1, output_2, output_3, self.embedded_words])
        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
        outp_list = []
        for i in range(config.model.global_att_num): 
            output_1 = keras.layers.Dense(units=output_.get_shape().as_list()[-1])(output_)
            output_1 = keras.layers.Dropout(rate=self.dropout_rate)(output_1)
            output_1 =  am.Scaled_Dot_Product_Att(output_.get_shape().as_list()[-1] // (config.model.global_att_num), name='multi_att_%d'%i)(output_1)
            outp_list.append(output_1)

        output_ = keras.layers.Concatenate()(outp_list)
#        output_ = keras.layers.MaxPool1D(pool_size=output_.get_shape().as_list()[-2], strides=1)(output_)
        output_ = keras.layers.Dense(units=output_.get_shape().as_list()[-1])(output_)
        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
        with tf.name_scope('output'):
            output_ = keras.layers.Flatten()(output_)
            output_ = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, activity_regularizer=keras.regularizers.l1_l2(l1=config.model.use_l1l2[0], l2=config.model.use_l1l2[1]))(output_)
            self.predictions = output_


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)

class CNN_LSTM(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
        
#        shap = self.embedded_words.get_shape().as_list()
#        output_ = keras.layers.Reshape([-1, shap[-1], 1])(self.embedded_words)
        
        output_ = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='relu')(self.embedded_words)
        shap = output_.get_shape().as_list()
        print(shap)
#        output_ = keras.layers.Reshape([-1, shap[-1] * shap[-2]])(output_)
        
        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
        output_ = keras.layers.CuDNNLSTM(units=128, return_sequences=True)(output_)

        
        output_ = keras.layers.Flatten()(output_)
        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)


class BiLSTM(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
        
 
        output_ = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True),
                                              merge_mode='concat')(self.embedded_words)
        output_ = keras.layers.Flatten()(output_)

        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)



class ShortBiLSTM(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
        
 
        output_ = am.ShortCNNLSTM(units=config.model.hidden_size, return_sequences=True, kernel_size=3, stride=3)(self.embedded_words)
        output_ = keras.layers.Flatten()(output_)

        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)



class AC_BiLSTM(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
        
 
        output_ = keras.layers.Conv1D(filters=100, kernel_size=3, strides=1)(self.embedded_words)
    
        output_1 = keras.layers.CuDNNLSTM(units=150, return_sequences=True)(output_)
        output_2 = keras.layers.CuDNNLSTM(units=150, return_sequences=True, go_backwards=True)(output_)
        
        output_1 = am.Attention(config.model.hidden_size_att)(output_1)
        output_2 = am.Attention(config.model.hidden_size_att)(output_2)
        output_= keras.layers.Concatenate()([output_1, output_2])
        output_ = keras.layers.Flatten()(output_)

        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)



class LSTM(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
        
        output_ = keras.layers.CuDNNLSTM(units=config.model.hidden_size * 2, return_sequences=False, go_backwards=False, name='LSTM')(self.embedded_words)
        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)

class RCNN(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
        
 
        output_1 = keras.layers.SimpleRNN(units=config.model.hidden_size, activation='tanh', return_sequences=True, go_backwards=False, name='cnn_forward')(self.embedded_words)
        output_2 = keras.layers.SimpleRNN(units=config.model.hidden_size, activation='tanh', return_sequences=True,go_backwards=True, name='cnn_back')(self.embedded_words)
        output_ = keras.layers.Concatenate(axis=-1)([output_1,self.embedded_words, output_2])
        output_ = keras.layers.Dense(units=config.model.hidden_size_att, activation='tanh', use_bias=True)(output_)
        output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)

        shap = output_.get_shape().as_list()
        output_ = keras.layers.MaxPool1D(pool_size=shap[-2], strides=1)(output_)
        
        output_= keras.layers.Flatten()(output_)
        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)

        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)



class Cov_max_pool(object):

    def __init__(self, config, word_embedding):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
#            self.embedded_words = keras.layers.SpatialDropout1D(rate=0.2)(self.embedded_words)
        output_list = []
        for i in range(3,6):
            output_ = keras.layers.Conv1D(filters=config.model.hidden_size, kernel_size=i, activation='tanh', padding='same')(self.embedded_words)
            output_ = keras.layers.GlobalMaxPool1D(data_format='channels_last')(output_)
            output_list.append(output_)
        
        output_ = keras.layers.Concatenate()(output_list)
#        output_ = keras.layers.Flatten()(output_)
#        output = keras.layers.Dropout(rate=1 - self.dropout_rate)(output)

        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax', activity_regularizer=keras.regularizers.l2(0.001))(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)



class Bilstm_Att(object):

    def __init__(self, config, word_embedding,hidden_size=128):
#        self.input_X = keras.layers.Input(shape=(config.word.sentence_len,), batch_size=config.model.batchsize, dtype='int64', name="input_X")
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)
#            self.embedded_words = keras.layers.SpatialDropout1D(rate=0.2)(self.embedded_words)
        with tf.name_scope('bi_lstm'):
            lstm_ = keras.layers.Bidirectional(keras.layers.LSTM(units=config.model.hidden_size, return_sequences=True),
                                                  merge_mode='concat')(self.embedded_words)
            output_ = keras.layers.Dropout(rate=self.dropout_rate)(lstm_)

        with tf.name_scope('attention'):
            output_ = am.SelfAtt(config.model.hidden_size)(output_)
            output_ = keras.layers.Flatten()(output_)

        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)

            self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)


class Stacked_Bilstm_038(object):

    def __init__(self, config, word_embedding):
        self.input_word = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_word")
        self.rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_word)

        with tf.name_scope('stacked_bilstm'):
            output_1 = keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True, go_backwards=False, name='lstm_1')(self.embedded_words)
            output_1 = keras.layers.Dropout(rate=self.rate)(output_1)
            output_1 = keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=False,go_backwards=False, name='lstm_2')(output_1)
            output_2 = keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True,go_backwards=True, name='lstm_b_1')(self.embedded_words)
            output_2 = keras.layers.Dropout(rate=self.rate)(output_2)
            output_2 = keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=False,go_backwards=True, name='lstm_b_2')(output_2)
            output_ = keras.layers.Concatenate(axis=-1)([output_1, output_2])

        with tf.name_scope('output'):
            self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)

            self.model = keras.Model(inputs=self.input_word, outputs=self.predictions)

class BIGRU_CNN_023(object):

    def __init__(self, config, word_embedding):
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)

        with tf.name_scope('bi_GRU'):
            lstm_ = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=config.model.hidden_size, return_sequences=True),
                                                  merge_mode='concat',name='bi_GRU')(self.embedded_words)
            output_ = keras.layers.Dropout(rate=self.dropout_rate)(lstm_)

        with tf.name_scope('cnn_POOl'):
#            shap = output_.get_shape().as_list()
#            output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
            output_ = keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=True,
                                              padding='same', activation=None,
                                              kernel_initializer='he_normal', bias_initializer ='zeros',
                                              name='conv_1')(output_)
            output_= am.HLU(0.1)(output_)
            # output_ = keras.layers.LeakyReLU(0.03)(output_)
            # output_ = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(output_)
            output_ = keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(output_)
            output_ = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, use_bias=True,
                                                          padding='same', activation=None,
                                                          kernel_initializer='he_normal', bias_initializer ='zeros',
                                                          name='conv_2')(output_)
            # output_ = keras.layers.LeakyReLU(0.03)(output_)
            # output_ = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(output_)
            output_= am.HLU(0.1)(output_)
            output_ = keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(output_)

            #
            output_ = keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, use_bias=True,
                                          padding='same', activation=None,
                                          kernel_initializer='he_normal', bias_initializer='zeros',
                                          name='conv_3')(output_)
            # output_ = keras.layers.LeakyReLU(0.3)(output_)
            output_ = am.HLU(0.1)(output_)
            # output_ = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(output_)
            output_ = keras.layers.AveragePooling1D(pool_size=1, strides=None, padding='valid', data_format='channels_last')(
                output_)
            output_ = keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, use_bias=True,
                                          padding='same', activation=None,
                                          kernel_initializer='he_normal', bias_initializer='zeros',
                                          name='conv_4')(output_)


            # output_ = keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(output_)
            # output_ = keras.layers.LeakyReLU(0.03)(output_)
            output_ = am.HLU(0.1)(output_)
            output_ = keras.layers.AveragePooling1D(pool_size=1, strides=None, padding='valid', data_format='channels_last')(
                output_)
            output_ = am.SelfAtt(config.model.hidden_size)(output_)
            output_ = keras.layers.GlobalAveragePooling1D('channels_last')(output_)

#            output_ = keras.layers.Flatten()(output_)
#            output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)

        with tf.name_scope('output'):
            # self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax')(output_) 原全连接层和输出层
            # output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)
            output_ = keras.layers.Dense(units=config.label_len, activation=None)(output_)
            output_ = keras.layers.Reshape((28,8))(output_)
            self.predictions = keras.layers.Softmax(axis=-1)(output_)


        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)

        def get_flops(model):
            run_meta = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(graph=tf.keras.backend.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops
        #浮点数Flops计算

        self.model.summary()
        FLOPs = get_flops(self.model)
        print(FLOPs)


class BILSTM_CNN_023(object):

    def __init__(self, config, word_embedding):
        self.input_X = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_X")
        self.dropout_rate = config.model.dropout_rate

        with tf.name_scope('embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_X)

        with tf.name_scope('bi_GRU'):
            lstm_ = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True),
                                                  merge_mode='concat')(self.embedded_words)
            output_ = keras.layers.Dropout(rate=self.dropout_rate)(lstm_)

        with tf.name_scope('cnn_POOl'):
#            shap = output_.get_shape().as_list()
#            output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
            output_ = keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, use_bias=True,
                                              padding='same', activation=None,
                                              kernel_initializer='he_normal', bias_initializer ='zeros',
                                              name='conv_1')(output_)
#            output_ = keras.layers.LeakyReLU(0.01)(output_)
            output_= am.HLU(0.1)(output_)
            output_ = keras.layers.GlobalAveragePooling1D('channels_last')(output_)
#            output_ = keras.layers.Flatten()(output_)
#            output_ = keras.layers.Dropout(rate=self.dropout_rate)(output_)

        with tf.name_scope('output'):
            if isinstance(config.model.use_l1l2, list) and len(config.model.use_l1l2) == 2:
                print('use_l1l2')
                self.predictions = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, activity_regularizer=keras.regularizers.l1_l2(l1=config.model.use_l1l2[0], l2=config.model.use_l1l2[1]))(output_)
            else:
                print('not use_l1l2')
                self.predictions = keras.layers.Dense(units=config.label_len, use_bias=True, activation='softmax')(output_)

        self.model = keras.Model(inputs=self.input_X, outputs=self.predictions)


class DMC_DCRNET_ATT(object):
    def __init__(self, config, word_embedding):
        self.input_word = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_word")
        self.rate = config.model.dropout_rate

        with tf.name_scope('word_embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_word)
            if config.model.spa_drop_rate != 0.0:
                self.embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(self.embedded_words)

        with tf.name_scope('word_cnn'):

            shap = self.embedded_words.get_shape().as_list()
            output_list=[]
            output_list.append(self.embedded_words)
            for i in range(1, 1 + config.model.multi_channel_num):
                output_1 = keras.layers.Conv1D(filters=config.model.hidden_size*2,kernel_size=3, strides=1, padding='same',use_bias=True,
                                                kernel_initializer='he_normal', bias_initializer ='zeros',
                                                activation=None,dilation_rate=i, name='conv_rate_%d'%i)(self.embedded_words)
                output_1 =  am.HLU(0.15)(output_1)
                output_1 = keras.layers.Dropout(rate=self.rate)(output_1)
                output_list.append(output_1)

        with tf.name_scope('bi_lstm_word'):
            for i in range(len(output_list)):
                output_ = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True),
                                                      merge_mode='concat', name='bi_lstm_word_%d'%i)(output_list[i])
                output_ = keras.layers.Dropout(rate=self.rate, name='bi_lstm_word_dropout_%d'%i)(output_)
                output_ =  am.SelfAtt(config.model.hidden_size_att, use_weighted=config.model.add_use_weighted, name='self_att_bilstm_%d'%i)(output_)
                output_list[i] = output_
            if len(output_list) > 1:
                output_ = keras.layers.Add(name='lstm_content_Add')(output_list)
            else:
                output_ = output_list[0]
        with tf.name_scope("Attention"):
            if config.model.global_att_num >= 1:
                output_ = keras.layers.Dropout(rate=self.rate)(output_)
                outp_list = []
                att_list = []
                shap = output_.get_shape().as_list()
                for i in range(config.model.global_att_num):
                    output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
                    output_1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=False,
                                                      padding='same', activation=None, dilation_rate=(1,1),
                                                      kernel_initializer='he_normal', bias_initializer ='zeros',
                                                      name='multi_att_conv_%d'%i)(output_)
                    output_1 =  am.Scaled_Dot_Product_Att(config.model.hidden_size // (i + 1), name='multi_att_%d'%i)(output_1)
                    att_list.append(config.model.hidden_size // (i + 1))
                    outp_list.append(output_1)
                if config.model.use_att_reg:
                    output_ =  am.MConcat_Reg(activity_regularizer= am.Reg1(config.model.use_att_reg, att_list))(outp_list)
                    print('use_att_reg')
                else:
                    if config.model.global_att_num > 1:
                        output_ = keras.layers.Concatenate(axis=-1)(outp_list)
                    else:
                        output_= outp_list[0]
                    print('not use_att_reg')
                    
            shap = output_.get_shape().as_list()                                                
            output_ = keras.layers.Lambda(lambda x: tf.transpose(tf.nn.top_k(tf.transpose(x,[0,2,1]),k=min(config.word.sentence_len, 10000 // shap[-1]),sorted=False)[0],[0,2,1]))(output_)
            output_ = keras.layers.Flatten(name='flatten')(output_)
            output_ = keras.layers.Dense(units=config.model.hidden_size//2, use_bias=True, activation='tanh')(output_)
            if isinstance(config.model.use_l1l2, list) and len(config.model.use_l1l2) == 2:
                print('use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, activity_regularizer=keras.regularizers.l1_l2(l1=config.model.use_l1l2[0], l2=config.model.use_l1l2[1]))(output_)
            else:
                print('not use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, use_bias=True, activation='softmax')(output_)
            self.predictions = output_

        self.model = keras.Model(inputs=[self.input_word], outputs=[self.predictions])



class DMC_DCRNET_ATT_1(object):
    def __init__(self, config, word_embedding):
        self.input_word = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_word")
        self.rate = config.model.dropout_rate

        with tf.name_scope('word_embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=config.model.mask_zero, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_word)
            if config.model.spa_drop_rate != 0.0:
                self.embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(self.embedded_words)

        with tf.name_scope('word_cnn'):

            shap = self.embedded_words.get_shape().as_list()
            output_list=[]
#            new_embedding = keras.layers.Reshape([-1, shap[-1], 1])(self.embedded_words)
            new_embedding = am.RemoveMask()(self.embedded_words)
            output_list.append(new_embedding)
            for i in range(1, 1 + config.model.multi_channel_num):
                output_1 = keras.layers.Conv1D(filters=config.model.hidden_size*2, kernel_size=config.model.kernel_size, strides=1, padding='same',use_bias=config.model.use_bias,
                                                kernel_initializer='he_normal', bias_initializer ='zeros',
                                                activation=None,dilation_rate=i, name='conv_rate_%d'%i)(new_embedding)
#                output_1 = keras.layers.Conv2D(filters=1,kernel_size=(3, 1), strides=(1,1), padding='same',use_bias=config.model.use_bias,
#                                                kernel_initializer='he_normal', bias_initializer ='zeros',
#                                                activation=None,dilation_rate=(i, 1), name='conv_rate_%d'%i)(new_embedding)
                output_1 =  am.HLU(0.15)(output_1)
                if config.model.mask_zero:
                    output_1 = am.RestoreMask(used=True)([output_1, self.embedded_words])
                output_1 = keras.layers.Dropout(rate=self.rate)(output_1)
                
                output_1 = am.RemoveMask()(output_1)
#                output_1 = keras.layers.Reshape([shap[-2], shap[-1]])(output_1)
                output_list.append(output_1)

        with tf.name_scope('bi_lstm_word'):
            for i in range(len(output_list)):
                output_ = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True),
                                                      merge_mode='concat', name='bi_lstm_word_%d'%i)(output_list[i])
                if config.model.mask_zero:
                    output_ = am.RestoreMask(used=True)([output_, self.embedded_words])
                output_ = keras.layers.Dropout(rate=self.rate, name='bi_lstm_word_dropout_%d'%i)(output_)
                output_ =  am.SelfAtt(config.model.hidden_size_att, use_weighted=config.model.add_use_weighted, return_probabilities=config.model.local_return_probabilities, name='self_att_bilstm_%d'%i)(output_)
                output_ = keras.layers.Dropout(rate=self.rate)(output_)
                output_ = am.RemoveMask()(output_)
                output_list[i] = output_
            if len(output_list) > 1:
                output_ = keras.layers.Add(name='lstm_content_Add')(output_list)
            else:
                output_ = output_list[0]
        with tf.name_scope("Attention"):
            if config.model.global_att_num >= 1:
                outp_list = []
                att_list = []
                shap = output_.get_shape().as_list()
                top_k_thro = 1
                for i in range(config.model.global_att_num):
                    output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
                    output_1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=True,
                                                      padding='same', activation=None, dilation_rate=(1,1),
                                                      kernel_initializer='he_normal', bias_initializer ='zeros',
                                                      name='multi_att_conv_%d'%i)(output_)
                    if config.model.mask_zero:
                        output_1 = am.RestoreMask(used=True)([output_1, self.embedded_words])
                    output_1 =  am.Scaled_Dot_Product_Att(config.model.hidden_size // (i + 1), return_probabilities=config.model.global_return_probabilities,name='multi_att_%d'%i)(output_1)
                    att_list.append(config.model.hidden_size // (i + 1))
                    top_k_thro += config.model.hidden_size // (i + 1)
                    outp_list.append(output_1)
                if config.model.use_att_reg:
                    output_ =  am.MConcat_Reg(activity_regularizer= am.Reg1(config.model.use_att_reg, att_list))(outp_list)
                    print('use_att_reg')
                else:
                    if config.model.global_att_num > 1:
                        output_ = keras.layers.Concatenate(axis=-1)(outp_list)
                    else:
                        output_= outp_list[0]
                    print('not use_att_reg')
                    
            shap = output_.get_shape().as_list()                                                
            output_ = am.Top_K(k=min(config.word.sentence_avg_len + config.word.sentence_len // 10, 10000 // top_k_thro), sortable=True)(output_)
            output_ = keras.layers.Flatten(name='flatten')(output_)
            output_ = keras.layers.Dense(units=config.model.hidden_size//2, use_bias=True, activation='tanh')(output_)
            if isinstance(config.model.use_l1l2, list) and len(config.model.use_l1l2) == 2:
                print('use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, activity_regularizer=keras.regularizers.l1_l2(l1=config.model.use_l1l2[0], l2=config.model.use_l1l2[1]))(output_)
            else:
                print('not use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, use_bias=True, activation='softmax')(output_)
            self.predictions = output_

        self.model = keras.Model(inputs=[self.input_word], outputs=[self.predictions])


class DMC_DCRNET_ATT_GRU(object):
      def __init__(self, config, word_embedding):
        self.input_word = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_word")
        self.rate = config.model.dropout_rate

        with tf.name_scope('word_embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=True, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_word)
            if config.model.spa_drop_rate != 0.0:
                self.embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(self.embedded_words)

        with tf.name_scope('word_cnn'):

            shap = self.embedded_words.get_shape().as_list()
            output_list=[]
#            new_embedding = keras.layers.Reshape([-1, shap[-1], 1])(self.embedded_words)
            new_embedding = am.RemoveMask()(self.embedded_words)
            output_list.append(new_embedding)
            for i in range(1, 1 + config.model.multi_channel_num):
                output_1 = keras.layers.Conv1D(filters=config.model.hidden_size*2, kernel_size=config.model.kernel_size, strides=1, padding='same',use_bias=config.model.use_bias,
                                                kernel_initializer='he_normal', bias_initializer ='zeros',
                                                activation=None,dilation_rate=i, name='conv_rate_%d'%i)(new_embedding)
#                output_1 = keras.layers.Conv2D(filters=1,kernel_size=(3, 1), strides=(1,1), padding='same',use_bias=config.model.use_bias,
#                                                kernel_initializer='he_normal', bias_initializer ='zeros',
#                                                activation=None,dilation_rate=(i, 1), name='conv_rate_%d'%i)(new_embedding)
                output_1 =  am.HLU(0.15)(output_1)
                if config.model.mask_zero:
                    output_1 = am.RestoreMask(used=True)([output_1, self.embedded_words])
                output_1 = keras.layers.Dropout(rate=self.rate)(output_1)
                output_1 = am.RemoveMask()(output_1)
#                output_1 = keras.layers.Reshape([shap[-2], shap[-1]])(output_1)
                output_list.append(output_1)

        with tf.name_scope('bi_lstm_word'):
            for i in range(len(output_list)):
                output_ = keras.layers.Bidirectional(keras.layers.CuDNNGRU(units=config.model.hidden_size, return_sequences=True),
                                                      merge_mode='concat', name='bi_lstm_word_%d'%i)(output_list[i])
                if config.model.mask_zero:
                    output_ = am.RestoreMask(used=True)([output_, self.embedded_words])
                output_ = keras.layers.Dropout(rate=self.rate, name='bi_lstm_word_dropout_%d'%i)(output_)
                output_ =  am.SelfAtt(config.model.hidden_size_att, use_weighted=config.model.add_use_weighted, return_probabilities=config.model.local_return_probabilities, name='self_att_bilstm_%d'%i)(output_)
                output_ = keras.layers.Dropout(rate=self.rate)(output_)
                output_ = am.RemoveMask()(output_)
                output_list[i] = output_
            if len(output_list) > 1:
                output_ = keras.layers.Add(name='lstm_content_Add')(output_list)
            else:
                output_ = output_list[0]
        with tf.name_scope("Attention"):
            if config.model.global_att_num >= 1:
                outp_list = []
                att_list = []
                shap = output_.get_shape().as_list()
                top_k_thro = 1
                for i in range(config.model.global_att_num):
                    output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
                    output_1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=True,
                                                      padding='same', activation=None, dilation_rate=(1,1),
                                                      kernel_initializer='he_normal', bias_initializer ='zeros',
                                                      name='multi_att_conv_%d'%i)(output_)
                    if config.model.mask_zero:
                        output_1 = am.RestoreMask(used=True)([output_1, self.embedded_words])
                    output_1 =  am.Scaled_Dot_Product_Att(config.model.hidden_size // (i + 1), return_probabilities=config.model.global_return_probabilities,name='multi_att_%d'%i)(output_1)
                    att_list.append(config.model.hidden_size // (i + 1))
                    top_k_thro += config.model.hidden_size // (i + 1)
                    outp_list.append(output_1)
                if config.model.use_att_reg:
                    output_ =  am.MConcat_Reg(activity_regularizer= am.Reg1(config.model.use_att_reg, att_list))(outp_list)
                    print('use_att_reg')
                else:
                    if config.model.global_att_num > 1:
                        output_ = keras.layers.Concatenate(axis=-1)(outp_list)
                    else:
                        output_= outp_list[0]
                    print('not use_att_reg')
                    
            shap = output_.get_shape().as_list()                                                
            output_ = am.Top_K(k=min(config.word.sentence_avg_len + config.word.sentence_len // 10, 10000 // top_k_thro), sortable=True)(output_)
            output_ = keras.layers.Flatten(name='flatten')(output_)
            output_ = keras.layers.Dense(units=config.model.hidden_size//2, use_bias=True, activation='tanh')(output_)
            if isinstance(config.model.use_l1l2, list) and len(config.model.use_l1l2) == 2:
                print('use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, activity_regularizer=keras.regularizers.l1_l2(l1=config.model.use_l1l2[0], l2=config.model.use_l1l2[1]))(output_)
            else:
                print('not use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, use_bias=True, activation='softmax')(output_)
            self.predictions = output_

        self.model = keras.Model(inputs=[self.input_word], outputs=[self.predictions])



class DMC_DCRNET_ATT_global_avg(object):
   def __init__(self, config, word_embedding):
        self.input_word = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_word")
        self.rate = config.model.dropout_rate

        with tf.name_scope('word_embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_word)
            if config.model.spa_drop_rate != 0.0:
                self.embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(self.embedded_words)

        with tf.name_scope('word_cnn'):

            shap = self.embedded_words.get_shape().as_list()
            output_list=[]
            output_list.append(self.embedded_words)
            for i in range(1, 1 + config.model.multi_channel_num):
                output_1 = keras.layers.Conv1D(filters=config.model.hidden_size*2,kernel_size=3, strides=1, padding='same',use_bias=True,
                                                kernel_initializer='he_normal', bias_initializer ='zeros',
                                                activation=None,dilation_rate=i, name='conv_rate_%d'%i)(self.embedded_words)
                output_1 =  am.HLU(0.15)(output_1)
                output_1 = keras.layers.Dropout(rate=self.rate)(output_1)
                output_list.append(output_1)

        with tf.name_scope('bi_lstm_word'):
            for i in range(len(output_list)):
                output_ = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True),
                                                      merge_mode='concat', name='bi_lstm_word_%d'%i)(output_list[i])
                output_ = keras.layers.Dropout(rate=self.rate, name='bi_lstm_word_dropout_%d'%i)(output_)
                output_ =  am.SelfAtt(config.model.hidden_size_att, name='self_att_bilstm_%d'%i)(output_)
                output_list[i] = output_
            if len(output_list) > 1:
                output_ = keras.layers.Add(name='lstm_content_add')(output_list)
            else:
                output_ = output_list[0]
            output_ = keras.layers.Dropout(rate=self.rate)(output_)
        with tf.name_scope("Attention"):
            if config.model.global_att_num >= 1:
                outp_list = []
                att_list = []
                shap = output_.get_shape().as_list()
                for i in range(config.model.global_att_num):
                    output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
                    output_1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=False,
                                                      padding='same', activation=None, dilation_rate=(1,1),
                                                      kernel_initializer='he_normal', bias_initializer ='zeros',
                                                      name='multi_att_conv_%d'%i)(output_)
                    output_1 =  am.Scaled_Dot_Product_Att(config.model.hidden_size // (i + 1), name='multi_att_%d'%i)(output_1)
                    att_list.append(config.model.hidden_size // (i + 1))
                    outp_list.append(output_1)
                if config.model.use_att_reg:
                    output_ =  am.MConcat_Reg(activity_regularizer= am.Reg1(config.model.use_att_reg, att_list))(outp_list)
                    print('use_att_reg')
                else:
                    if config.model.global_att_num > 1:
                        output_ = keras.layers.Concatenate(axis=-1)(outp_list)
                    else:
                        output_= outp_list[0]
                    print('not use_att_reg')
                shap = output_.get_shape().as_list()                                                
            output_ = keras.layers.GlobalAveragePooling1D()(output_)
            output_ = keras.layers.Dense(units=config.model.hidden_size//2)(output_)
            if config.model.use_l1l2:
                print('use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax', activity_regularizer=keras.regularizers.l1_l2(config.model.use_l1l2))(output_)
            else:
                print('not use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax')(output_)
            self.predictions = output_

        self.model = keras.Model(inputs=[self.input_word], outputs=[self.predictions])


class DMC_DCRNET_ATT_mean_k(object):
     def __init__(self, config, word_embedding):
        self.input_word = keras.layers.Input(batch_shape=(None, config.word.sentence_len,),  dtype='int64', name="input_word")
        self.rate = config.model.dropout_rate

        with tf.name_scope('word_embbeding'):
            self.W = keras.layers.Embedding(input_dim=len(word_embedding),output_dim=config.word.embedding_size,
                                                            input_length=config.word.sentence_len, weights=[word_embedding],
                                                            mask_zero=False, trainable=False, name='word_embbeding')

            self.embedded_words = self.W(self.input_word)
            if config.model.spa_drop_rate != 0.0:
                self.embedded_words = keras.layers.SpatialDropout1D(rate=config.model.spa_drop_rate)(self.embedded_words)

        with tf.name_scope('word_cnn'):

            shap = self.embedded_words.get_shape().as_list()
            output_list=[]
            output_list.append(self.embedded_words)
            for i in range(1, 1 + config.model.multi_channel_num):
                output_1 = keras.layers.Conv1D(filters=config.model.hidden_size*2,kernel_size=3, strides=1, padding='same',use_bias=True,
                                                kernel_initializer='he_normal', bias_initializer ='zeros',
                                                activation=None,dilation_rate=i, name='conv_rate_%d'%i)(self.embedded_words)
                output_1 =  am.HLU(0.15)(output_1)
#                output_1 = keras.layers.Dropout(rate=self.rate)(output_1)
                output_list.append(output_1)

        with tf.name_scope('bi_lstm_word'):
            for i in range(len(output_list)):
                output_ = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(units=config.model.hidden_size, return_sequences=True),
                                                      merge_mode='concat', name='bi_lstm_word_%d'%i)(output_list[i])
#                output_ = keras.layers.Dropout(rate=self.rate, name='bi_lstm_word_dropout_%d'%i)(output_)
                output_ =  am.SelfAtt(config.model.hidden_size_att, use_weighted=config.model.add_use_weighted, name='self_att_bilstm_%d'%i)(output_)
                output_list[i] = output_
            if len(output_list) > 1:
                output_ = keras.layers.Add(name='lstm_content_Add')(output_list)
            else:
                output_ = output_list[0]
        with tf.name_scope("Attention"):
            if config.model.global_att_num >= 1:
                output_ = keras.layers.Dropout(rate=self.rate)(output_)
                outp_list = []
                att_list = []
                shap = output_.get_shape().as_list()
                for i in range(config.model.global_att_num):
                    output_ = keras.layers.Reshape([-1, shap[-1], 1])(output_)
                    output_1 = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), use_bias=False,
                                                      padding='same', activation=None, dilation_rate=(1,1),
                                                      kernel_initializer='he_normal', bias_initializer ='zeros',
                                                      name='multi_att_conv_%d'%i)(output_)
                    output_1 =  am.Scaled_Dot_Product_Att(config.model.hidden_size // (i + 1), name='multi_att_%d'%i)(output_1)
                    att_list.append(config.model.hidden_size // (i + 1))
                    outp_list.append(output_1)
                if config.model.use_att_reg:
                    output_ =  am.MConcat_Reg(activity_regularizer= am.Reg1(config.model.use_att_reg, att_list))(outp_list)
                    print('use_att_reg')
                else:
                    if config.model.global_att_num > 1:
                        output_ = keras.layers.Concatenate(axis=-1)(outp_list)
                    else:
                        output_= outp_list[0]
                    print('not use_att_reg')
                    
            shap = output_.get_shape().as_list()                                                
            output_ = keras.layers.Lambda(lambda x: tf.transpose(tf.nn.top_k(tf.transpose(x,[0,2,1]),k=min(config.word.sentence_avg_len + config.word.sentence_len // 10, 10000 // shap[-1]),sorted=True)[0],[0,2,1]))(output_)
            output_ = keras.layers.GlobalAveragePooling1D()(output_)
            if len(output_.get_shape().as_list()) > 2:
                output_ = keras.layers.Flatten(name='flatten')(output_)

            output_ = keras.layers.Dense(units=config.model.hidden_size//2, use_bias=True, activation='tanh')(output_)
            if isinstance(config.model.use_l1l2, list) and len(config.model.use_l1l2) == 2:
                print('use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, activation='softmax', use_bias=True, activity_regularizer=keras.regularizers.l1_l2(l1=config.model.use_l1l2[0], l2=config.model.use_l1l2[1]))(output_)
            else:
                print('not use_l1l2')
                output_ = keras.layers.Dense(units=config.label_len, use_bias=True, activation='softmax')(output_)
            self.predictions = output_

        self.model = keras.Model(inputs=[self.input_word], outputs=[self.predictions])


class TextClassifier(object):
    
    def __init__(self, model_name, config, word_embedding, summary_enabled = True):
        if model_name == 'default' or  model_name == 'DMC_DCRNET_ATT':
            self.model = DMC_DCRNET_ATT(config, word_embedding).model
        elif model_name == 'BIGRU_CNN_023':
            self.model = BIGRU_CNN_023(config, word_embedding).model
        elif model_name == 'BILSTM_CNN_023':
            self.model = BILSTM_CNN_023(config, word_embedding).model
        elif model_name == 'Stacked_Bilstm_038':
            self.model = Stacked_Bilstm_038(config, word_embedding).model
        elif model_name == 'Bilstm_Att':
            self.model = Bilstm_Att(config, word_embedding).model
        elif model_name == 'Cov_max_pool':
            self.model = Cov_max_pool(config, word_embedding).model
        elif model_name == 'DMC_DCRNET_ATT_global_avg':
            self.model = DMC_DCRNET_ATT_global_avg(config, word_embedding).model
        elif model_name == 'DMC_DCRNET_ATT_1':
            self.model = DMC_DCRNET_ATT_1(config, word_embedding).model      
        elif model_name == 'DMC_DCRNET_ATT_GRU':
            self.model = DMC_DCRNET_ATT_GRU(config, word_embedding).model      
        elif model_name == 'DMC_DCRNET_ATT_mean_k':
            self.model = DMC_DCRNET_ATT_mean_k(config, word_embedding).model
        elif model_name == 'RCNN':
            self.model = RCNN(config, word_embedding).model
        elif model_name == 'LSTM':
            self.model = LSTM(config, word_embedding).model
        elif model_name == 'BiLSTM':
            self.model = BiLSTM(config, word_embedding).model        
        elif model_name == 'ShortBiLSTM':
            self.model = ShortBiLSTM(config, word_embedding).model          
        elif model_name == 'CNN_LSTM':
            self.model = CNN_LSTM(config, word_embedding).model
        elif model_name == 'AC_BiLSTM':
            self.model = AC_BiLSTM(config, word_embedding).model
        elif model_name == 'ConV':
            self.model = ConV(config, word_embedding).model
        elif model_name == 'SConv':
            self.model = SConv(config, word_embedding).model
        elif model_name == 'TextBiLSTM_Attention':
            self.model = TextBiLSTM_Attention(config, word_embedding).model
        elif model_name == 'BiLSTMAttention':
            self.model = BiLSTMAttention(config, word_embedding)
        else:
            return
        
        # if summary_enabled is True:
        #     self.model.summary() #模型层数统计


    def fit(self, train_x, train_y, batch_size=32, epochs=1, callbacks=None, shuffle=True, class_weight=None, sample_weight=None, verbose=1, validation_split=0.2, validation_data=None):
        history = self.model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight, verbose=verbose, validation_split=validation_split, validation_data=validation_data)
        return history

    def prediction(self, text_x, batch_size=32):
        history = self.model.predict(text_x, batch_size=batch_size)
        return history


    def evaluate(self, x, y, batch_size=32):
        history = self.model.evaluate(x, y, batch_size=batch_size)
        return history

    def gen_metrics(self, true_Y, pred_Y, average_type='macro'):
        """
        生成acc和auc值
        """
        if true_Y.shape[-1] > 2:           
            average_type='macro'
        else:
            average_type='binary'
        print('use %s average'%average_type)
        true_Y = np.reshape(true_Y , (-1, 7))
        pred_Y = np.reshape(pred_Y , (-1, 7))
        true_label_Y  =  np.argmax(true_Y,axis=-1)
        pred_label_Y =  np.argmax(pred_Y, axis=-1)
        accuracy = sm.accuracy_score(np.argmax(true_Y,axis=-1), np.argmax(pred_Y,axis=-1))
        precision = sm.precision_score(np.argmax(true_Y,axis=-1), np.argmax(pred_Y,axis=-1),average=average_type)
        recall = sm.recall_score(np.argmax(true_Y,axis=-1), np.argmax(pred_Y,axis=-1),average=average_type)
        f1_score = sm.f1_score(np.argmax(true_Y,axis=-1), np.argmax(pred_Y,axis=-1),average=average_type)
        return  round(precision, 7), round(recall, 7), round(f1_score, 7) ,round(accuracy, 7) #


    def gen_weight(self, true_Y, pred_Y, labels=None):
        """
        生成acc和auc值
        """
        true_Y = np.reshape(true_Y , (-1, 7))
        pred_Y = np.reshape(pred_Y , (-1, 7))

        confusion = sm.confusion_matrix(true_Y, pred_Y, labels=labels)
        each_acc = np.nan_to_num(truediv(np.diag(confusion), np.sum(confusion, axis=1)))
        each_acc = np.exp(1 - each_acc)
        each_acc = each_acc / min(each_acc)
        return each_acc

    def get_confusion(self, true_y, pred_y, labels=None):
        true_y = np.reshape(true_y , (-1, 7))
        pred_y = np.reshape(pred_y , (-1, 7))
        confusion = sm.confusion_matrix(np.argmax(true_y,axis=-1), np.argmax(pred_y,axis=-1), labels=labels)
        print(confusion)
    
    def save(self, filepath):
        print('save path', filepath)
        self.model.save(filepath)


class TextBiLSTM_Attention(object):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 l2_reg_lambda=0.0):
        # 定义需要用户输入的placeholder
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')  # 定义为placeholder是为了实现lr递减

        l2_loss = tf.constant(0.0)

        # Embedding层
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                 name='W', trainable=True)
            # [batch_size, sequence_length, embedding_size]
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.input_x)

        # biLSTM层
        with tf.name_scope('biLSTM'):
            lstm_fw_cell = tf.keras.layers.LSTMCell(hidden_size)
            lstm_bw_cell = tf.keras.layers.LSTMCell(hidden_size)
            embedded_words_list = tf.unstack(self.embedded_words, sequence_length, axis=1)
            outputs, states_fw, states_bw = rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                                         lstm_bw_cell,
                                                                         embedded_words_list,
                                                                         dtype=tf.float32)
            self.outputs = tf.stack(outputs, axis=1)

        with tf.name_scope('attention'):
            self.W_attention = tf.get_variable(shape=[hidden_size * 2, hidden_size * 2],
                                               initializer=tf.random_normal_initializer(stddev=0.1),
                                               name='W_attention')
            self.b_attention = tf.get_variable(shape=[hidden_size * 2], name='b_attention')
            self.context_vector = tf.get_variable("what_is_the_informative_word",
                                                  shape=[hidden_size * 2],
                                                  initializer=tf.random_normal_initializer(stddev=0.1))
            # [batch_size*sequence_length, hidden_size*2]
            hidden_state = tf.reshape(self.outputs, [-1, hidden_size * 2])
            hidden_representation = tf.nn.tanh(tf.matmul(hidden_state, self.W_attention) + self.b_attention)
            hidden_representation = tf.reshape(hidden_representation, shape=[-1, sequence_length, hidden_size * 2])
            # 计算相似度
            hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vector)
            attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
            # 为了防止softmax溢出，所以用logits减去max，再进行softmax
            attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
            p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
            p_attention_expanded = tf.expand_dims(p_attention, axis=2)
            # 加权求和得到表示句子的向量
            sentence_representation = tf.multiply(p_attention_expanded, self.outputs)
            sentence_representation = tf.reduce_sum(sentence_representation, axis=1)

        with tf.name_scope('dropout'):
            # dropout防止过拟合
            self.rnn_drop = tf.nn.dropout(sentence_representation, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable(shape=[hidden_size * 2, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # 将dropout的输出乘以w再加b
            self.scores = tf.nn.xw_plus_b(self.rnn_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope('loss'):
            # 交叉熵loss
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            # L2正则化后的loss
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
# 用于产生batch
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        data_size = len(data)
        num_batches_per_epoch = data_size// batch_size # 每个epoch中包含的batch数量
        for epoch in range(num_epochs):
            # 每个epoch是否进行shuflle
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data

            for batch_num in range(num_batches_per_epoch+1):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


import tensorflow as tf
import keras
import sklearn.metrics as sm
import numpy as np
from operator import truediv
from data_config_preprocess import Config
import attention_method as am
# 构建模型


class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, word_embedding,dtype='int64',hidden_size = 128):

        # 定义模型的输入
        self.config = config
        self.word_embedding = word_embedding
        self.inputX = tf.placeholder(tf.int32, [None, 17], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(word_embedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hidden_size):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H)
            outputSize = config.model.hidden_size[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")

            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.config.model.hidden_size[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, 17])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, 17, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output


a = [1,2,3,4,5,6,7,0]
b = [1,2,3,4,4,4,4,0]
sm.accuracy_score(a,b)