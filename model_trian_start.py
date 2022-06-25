# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:04:48 2019
 层连
@author: Administrator
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import keras


    
    
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical
import numpy as np
#import langconv
import os
import sys
import time
import warnings
import gc
from attention_method import precision, recall, f1
from data_config_preprocess import Dataset, Config
from textclassifier_m import TextClassifier
from collections import Counter
import argparse


warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='tensorflow')
    
def main():
    
    config = Config()
    data = Dataset(config)
    data.data_gen()

    train_contents = data.train_content_word
    train_labels = data.train_label
    valid_contents = data.valid_content_word
    valid_labels = data.valid_label
    test_contents = data.test_content_word
    test_labels = data.test_label

    
    word_emb = np.asarray(data.word_embedding)
    print(word_emb)
    print(train_contents)
    del data
    gc.collect()

    
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.99
    sess = tf.Session(config=session_conf)
    keras.backend.set_session(sess)
    
    
    callback=[]
    save_best_model = keras.callbacks.ModelCheckpoint(filepath='./me_model/%s.h5'%config.save_model_name, monitor='val_loss', verbose=1,
                                  save_best_only=True, mode='auto')

#    if config.model.use_early_stop > 0:
##        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='max')
#        early_stopping = Check_Early_Stop(config)
#        callback.append(early_stopping)
#        print('use early stop')
#    elif config.model.use_liner_delay_epoch > 0:
#        delay_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=config.model.learning_rate_ld_decay, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=config.model.learning_rate_ld_min)
#        callback.append(delay_lr)
#        print('use early stop')

    # valid_contents, valid_labels = valid_contents.split(), to_categorical(valid_labels)
    # train_content, train_labels = train_contents, to_categorical(train_labels)
    # test_contents, test_labels = test_contents, to_categorical(test_labels)
    #原标签部分
    # valid_contents, valid_labels = valid_contents, valid_labels.split(' ')
    # train_content, train_labels = train_contents, train_labels.split(' ')
    # test_contents, test_labels = test_contents, test_labels.split(' ')

    train_content = train_contents
    valid_contents = valid_contents
    test_contents = test_contents

    v_labels=list()
    t1_labels=list()
    t2_labels=list()
    for line in valid_labels:
        line=line.split()
        v_labels.append(list(map(int, line)))
    for line in train_labels:
        line=line.split()
        t1_labels.append(list(map(int, line)))
    for line in test_labels:
        line=line.split()
        t2_labels.append(list(map(int, line)))

#标签去空格，转Int型
    valid_labels = v_labels
    train_labels = t1_labels
    test_labels = t2_labels
    maxlen = 0
    for i in range(len(valid_labels)):
        if len(valid_labels[i]) < 28:
            valid_labels[i] += ([7]*(28 - len(valid_labels[i])))  #空白位置补0填充
        else:
            valid_labels[i] = valid_labels[i][:28]
        maxlen = max(maxlen, len(valid_labels[i]))


    for i in range(len(train_labels)):
        if len(train_labels[i]) < 28:
            train_labels[i] += ([7]*(28 - len(train_labels[i])))
        else:
            train_labels[i] = train_labels[i][:28]
        maxlen = max(maxlen, len(train_labels[i]))


    for i in range(len(test_labels)):
        if len(test_labels[i]) < 28:
            test_labels[i] += ([7]*(28 - len(test_labels[i])))
        else:
            test_labels[i] = test_labels[i][:28]
        maxlen = max(maxlen, len(test_labels[i]))


    print(maxlen)
    valid_labels = np.stack([keras.utils.to_categorical(valid, num_classes=8)[:, 0:] for valid in valid_labels])  #标签总述，从0开始
    train_labels = np.stack([keras.utils.to_categorical(valid, num_classes=8)[:, 0:] for valid in train_labels])
    test_labels = np.stack([keras.utils.to_categorical(valid, num_classes=8)[:, 0:] for valid in test_labels])

    print(train_labels)

    m_optimizer = keras.optimizers.Adam(lr=config.model.learning_rate, beta_1=0.9, beta_2=0.999, decay=0)
    classifier = TextClassifier(model_name='BIGRU_CNN_023', config=config, word_embedding=word_emb, summary_enabled=True)
    classifier.model.compile(loss='categorical_crossentropy', optimizer=m_optimizer, metrics=['accuracy'])

    classifier.fit(train_content, train_labels, batch_size=config.model.batchsize,
                   epochs=100, callbacks=callback, shuffle=True, sample_weight=None,
                   class_weight=None, verbose=config.model.verbose, validation_data=[valid_contents, valid_labels])

    pre_labels = classifier.prediction(test_contents, batch_size=config.model.batchsize)
    # print(np.argmax(pre_labels,axis=-1))
    f = open("result.txt", "w")
    for x in np.argmax(pre_labels, axis=-1):
        print(x)
        print(x,file=f)
    f.close()
    print(np.argmax(pre_labels, axis=-1))
    precision, recall, f1_score, acc = classifier.gen_metrics(test_labels, pre_labels, config.model.metrics_average_tpye)
    info_1 = "acc: %.4f, precision: %.4f, recall: %.4f, f1_score:%.4f" % (acc, precision, recall, f1_score)
    print(info_1)
    classifier.get_confusion(test_labels, pre_labels)

    # def get_flops(model):
    #     run_meta = tf.RunMetadata()
    #     opts = tf.profiler.ProfileOptionBuilder.float_operation()
    #     flops = tf.profiler.profile(graph=K.get_seesion().graph,run_meta = run_meta,cmd='op',options=opts)
    #     return flops.total_float_ops
    # FLOPs = get_flops(model)
    # print(FLOPs)

#
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state = 32)
#
#     model_name = ['']
#    # [ 224,  320, 448, 512]
#     batchsizes = [224, 320, 448, 512]
#     m_name = 'DMC_DCRNET_ATT_1'
#   #  model_name = ['DMC_DCRNET_ATT_1']
# #    train_labels = np.argmax(train_labels, axis=-1)
#     for m_name in model_name:
#         print('-' * 20)
#         print(m_name)
#         print('-' * 20)
#         k = 0
#         for tarin, test in kfold.split(train_contents, train_labels):
#             callback=[]
#             k+=1
#             print("*" * 20)
#             print("flod", m_name, ':', k)
#             print("*" * 20)
#
#         #    else:
#         #        print('use binary_crossentropy' )
#         #        classifier.model.compile(loss='binary_crossentropy', optimizer=m_optimizer, metrics=['accuracy'])
#             last_acc = 0.94
#             i = 0;
#             config.model.class_weights = [1.0 for _ in range(config.label_len)]
#             while True:
#                 if i >= config.model.epoches:
#                     break
#                 i += 1
#                 if config.model.use_early_stop == i:
#             #        early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='max')
#     #                early_stopping = Check_Early_Stop(config)
#     #                callback.append(early_stopping)
#                     print('use early stop')
#                 elif config.model.use_liner_delay == i:
#                     def scheduler(epoch):
#                         lr = keras.backend.get_value(classifier.model.optimizer.lr)
#                         if config.model.learning_rate_ld_min >= lr:
#                             lr = config.model.learning_rate_ld_min
#                         keras.backend.set_value(classifier.model.optimizer.lr, lr * config.model.learning_rate_ld_decay)
#                         print("lr changed to {}".format(lr * config.model.learning_rate_ld_decay))
#                         return keras.backend.get_value(classifier.model.optimizer.lr)
#
#                     reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
#         #            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=config.model.learning_rate_ld_decay, patience=2, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=config.model.learning_rate_ld_min)
#                     callback.append(reduce_lr)
#                     print('use lr delay')
#
#                 if config.model.use_class_weight[0] !=0 and i >= config.model.use_class_weight[0]:
#                     class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(train_label, -1)), np.argmax(train_label, -1))
#                     class_weights = [min(c, np.sqrt(np.e)) for c in class_weights]
#                     if i > config.model.use_class_weight[0] and valid_contents is not None:
#                         pre_labels = classifier.prediction(valid_contents, batch_size=config.model.batchsize)
#                         each_weight = classifier.gen_weight(valid_labels, pre_labels, list(range(config.label_len)))
#                     else:
#                         each_weight = [1 for _ in range(config.label_len)]
#         #            each_weight = [min(c, np.sqrt(np.e)) for c in each_weight]
#                     if len(config.model.use_class_weight) == 2 and i > config.model.use_class_weight[0]:
#                         a = config.model.use_class_weight[-1]
#                     else:
#                         a = 0
#                     class_weights = list(a + (1-a) * np.array(each_weight) * np.array(class_weights))
#                     class_weights = dict(zip([x for x in np.unique(np.argmax(train_label, -1))], class_weights))
#                 else:
#                     print("not use class weight")
#                     class_weights=None
#
#                 print('index:', i)
#                 classifier.fit(train_content, train_label, batch_size=config.model.batchsize,
#                                epochs=1, callbacks=callback, shuffle=True, sample_weight=None,
#                                class_weight=class_weights, verbose=config.model.verbose, validation_split=None)
#
#
#             pre_labels = classifier.prediction(test_contents, batch_size=config.model.batchsize)
#             if len(test_labels.shape) != 2:
#                 test_labels = to_categorical(test_labels)
#             precision, recall, f1_score, acc = classifier.gen_metrics(test_labels, pre_labels, config.model.metrics_average_tpye)
#             info_1 = "acc: %.4f, precision: %.4f, recall: %.4f, f1_score:%.4f"%(acc, precision, recall, f1_score)
#             print('\n index:', i, '------test info : ', info_1)
#             classifier.get_confusion(test_labels, pre_labels)
# #                    if (acc > last_acc):
# #                        last_acc = acc
# #                        classifier.save('./model/%s.h5'%config.save_model_name)
#             del classifier
#             gc.collect()
#
#
 #   pre_labels = classifier.prediction(test_contents, batch_size=config.model.batchsize)
#    classifier.get_confusion(test_labels, pre_labels)    
    
#    classifier.save('./model/%s.h5'%config.save_model_name)
#    classifier.model.save_weights('.\me_model\%s.weights' %config.save_model_name)


if __name__ == '__main__':
    main()
    # print(sys.argv)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-sl', '--sentence_len', type=int, nargs='?',
    #                     default=33,
    #                     help='the value of sentence_len')
    # parser.add_argument('-sal', '--sentence_avg_len', type=int, nargs='?',
    #                     default=10,
    #                     help='the value of sentence_avg_len')
    # parser.add_argument('-ws', '--word2vec_source', type=str, nargs='?',
    #                     default='',
    #                     help='the path of word2vec_source')
    # parser.add_argument('-trp', '--trainset_path', type=str, nargs='?',
    #                     default=None,
    #                     help='the path of  trainset_path_content and trainset_path_labels')
    # parser.add_argument('-vlp', '--valset_path', type=str, nargs='?',
    #                     default=None,
    #                     help='the path of  valset_path and trainset_path_labels')
    # parser.add_argument('-tep', '--testset_path', type=str, nargs='?',
    #                     default=None,
    #                     help='the path of  test_path_content and test_path_labels')
    # parser.add_argument('-vop', '--vocab_path', type=str, nargs='?',
    #                     default=None,
    #                     help='the path of  vocab_path')
    # parser.add_argument('-ll', '--label_len', type=int, nargs='?',
    #                     default=4,
    #                     help='the value of label_len')
    # parser.add_argument('-hz', '--hidden_size', type=int, nargs='?',
    #                     default=256,
    #                     help='the value of hidden_size')
    # parser.add_argument('-ut', '--use_tokenizer', type=int, nargs='?',
    #                     default=1,
    #                     help='the bool of use_tokenizer')
    # parser.add_argument('-hza', '--hidden_size_att', type=int, nargs='?',
    #                     default=128,
    #                     help='the value of hidden_size_att')
    # parser.add_argument('-uar', '--use_att_reg', type=float, nargs='?',
    #                     default=1.0,
    #                     help='the highparamater of use_att_reg')
    # parser.add_argument('-mcn', '--multi_channel_num', type=int, nargs='?',
    #                     default=3,
    #                     help='the value of multi_channel_num')
    # parser.add_argument('-gan', '--global_att_num', type=int, nargs='?',
    #                     default=3,
    #                     help='the value of global_att_num')
    # parser.add_argument('-mz', '--mask_zero', type=int, nargs='?',
    #                     default=1,
    #                     help='the mask_zero')
    # parser.add_argument('-gn', '--gpu_name', type=str, nargs='?',
    #                     default='0',
    #                     help='the name of gpu_name')
    # parser.add_argument('-mn', '--model_name', type=str, nargs='?',
    #                     default='default',
    #                     help='the name of model_name')
    # parser.add_argument('-utn', '--use_test_num', type=int, nargs='?',
    #                     default=0,
    #                     help='the value of use_test_num')
    # parser.add_argument('-uvn', '--use_valid_num', type=float, nargs='?',
    #                     default=0.0,
    #                     help='the value or percent of use_valid_num')
    # parser.add_argument('-bs', '--batchsize', type=int, nargs='?',
    #                     default=64,
    #                     help='the batchsize of training')
    # parser.add_argument('-ehs', '--epoches', type=int, nargs='?',
    #                     default=50,
    #                     help='the epoches of training')
    # parser.add_argument('-dr', '--dropout_rate', type=float, nargs='?',
    #                     default=0.5,
    #                     help='the dropout_rate of training')
    # parser.add_argument('-sdr', '--spa_drop_rate', type=float, nargs='?',
    #                     default=0.15,
    #                     help='the value of spa_drop_rate')
    # parser.add_argument('-lr', '--learning_rate', type=float, nargs='?',
    #                     default=0.001,
    #                     help='the value of learning_rate')
    # parser.add_argument('-uld', '--use_liner_delay', type=float, nargs='?',
    #                     default=0.001,
    #                     help='the value of use_liner_delay')
    # parser.add_argument('-ulde', '--use_liner_delay_epoch', type=int, nargs='?',
    #                     default=5,
    #                     help='the value of use_liner_delay_epoch')
    # parser.add_argument('-lrlmi', '--learning_rate_ld_min', type=float, nargs='?',
    #                     default=0.0001,
    #                     help='the value of learning_rate_ld_min')
    # parser.add_argument('-lrld', '--learning_rate_ld_decay', type=float, nargs='?',
    #                     default=0.98,
    #                     help='the value of learning_rate_ld_decay')
    # parser.add_argument('-lrlb', '--learning_rate_ld_batches', type=int, nargs='?',
    #                     default=5,
    #                     help='the value of learning_rate_ld_batches')
    # parser.add_argument('-utd', '--use_tri_delay', type=int, nargs='?',
    #                     default=1,
    #                     help='the bool of use_tri_delay')
    # parser.add_argument('-utde', '--use_tri_delay_epoch', type=int, nargs='?',
    #                     default=5,
    #                     help='the value of use_tri_delay_epoch')
    # parser.add_argument('-lrtma', '--learning_rate_td_max', type=float, nargs='?',
    #                     default=0.001,
    #                     help='the value of learning_rate_td_max')
    # parser.add_argument('-lrtmi', '--learning_rate_td_min', type=float, nargs='?',
    #                     default=0.0001,
    #                     help='the value of learning_rate_td_min')
    # parser.add_argument('-lrtd', '--learning_rate_td_decay', type=float, nargs='?',
    #                     default=0.98,
    #                     help='the value of learning_rate_td_decay')
    # parser.add_argument('-lrtb', '--learning_rate_td_batches', type=int, nargs='?',
    #                     default=5,
    #                     help='the value of learning_rate_td_batches')
    # parser.add_argument('-lrtf', '--learning_rate_td_frequency', type=int, nargs='?',
    #                     default=5,
    #                     help='the value of learning_rate_td_frequency')
    # parser.add_argument('-ucw', '--use_class_weight', type=float, nargs='*',
    #                     default=1.0,
    #                     help='the value of use_class_weight')
    # parser.add_argument('-ul1l2', '--use_l1l2', type=float, nargs='*',
    #                     default=None,
    #                     help='the value of use_l1l2')
    # parser.add_argument('-ues', '--use_early_stop', type=int, nargs='?',
    #                     default=1,
    #                     help='the max_delay_times of use_early_stop')
    # parser.add_argument('-ess', '--early_stop_step', type=int, nargs='?',
    #                     default=1,
    #                     help='the value of early_stop_step')
    # parser.add_argument('-esd', '--early_stop_delay', type=float, nargs='?',
    #                     default=0.8,
    #                     help='the value of early_stop_delay')
    # parser.add_argument('-esr', '--early_stop_reset', type=int, nargs='?',
    #                     default=10,
    #                     help='the value of early_stop_reset')
    # parser.add_argument('-smn', '--save_model_name', type=str, nargs='?',
    #                     default='default',
    #                     help='the name of save_model_name')
    # parser.add_argument('-pinfo', '--print_info', type=str, nargs='*',
    #                     default='',
    #                     help='the print text of info')
    # parser.add_argument('-vb', '--verbose', type=int, nargs='?',
    #                     default=1,
    #                     help='the verbose')
    # parser.add_argument('-ub', '--use_bias', type=int, nargs='?',
    #                     default=1,
    #                     help='the verbose')
    # parser.add_argument('-ks', '--kernel_size', type=int, nargs='?',
    #                     default=2,
    #                     help='the kernel_size')
    #
    #
    #
    # args = parser.parse_args()
    # config = Config()
    # config.word.sentence_len = args.sentence_len
    # print("set word.sentence_len", config.word.sentence_len)
    # config.word.sentence_avg_len = args.sentence_avg_len
    # print("set word.sentence_avg_len", config.word.sentence_avg_len)
    # config.word.word2vec_source = args.word2vec_source
    # print("set word.word2vec_source", config.word.word2vec_source)
    # config.trainset_path=args.trainset_path
    # print("set trainset_path", config.trainset_path)
    # config.testset_path=args.testset_path
    # print("set testset_path", config.testset_path)
    # config.vocab_path=args.vocab_path
    # print("set vacab_path", config.vocab_path)
    #
    # config.label_len = args.label_len
    # print("set label_len", config.label_len)
    # config.use_tokenizer = args.use_tokenizer
    # print("set use_tokenizer", config.use_tokenizer)
    #
    # config.model.kernel_size = args.kernel_size
    # print("set kernel_size", config.model.kernel_size)
    #
    # config.model.hidden_size = args.hidden_size
    # print("set model.hidden_size", config.model.hidden_size)
    #
    # config.model.hidden_size_att = args.hidden_size_att
    # print("set model.hidden_size_att", config.model.hidden_size_att)
    #
    # config.model.use_att_reg = args.use_att_reg
    # print("set model.use_att_reg", config.model.use_att_reg)
    #
    # config.model.multi_channel_num = args.multi_channel_num
    # print("set model.multi_channel_num", config.model.multi_channel_num)
    #
    # config.model.global_att_num = args.global_att_num
    # print("set model.global_att_num", config.model.global_att_num)
    #
    # config.model.mask_zero = bool(args.mask_zero)
    # print("set mask_zero", config.model.mask_zero)
    #
    # config.gpu_name = args.gpu_name
    # print("set gpu_name", config.gpu_name)
    #
    # config.model.model_name = args.model_name
    # print("set model.model_name", config.model.model_name)
    #
    # config.model.use_test_num = args.use_test_num
    # print("set model.use_test_num", config.model.use_test_num)
    #
    # config.model.use_valid_num = args.use_valid_num
    # if config.model.use_valid_num > 1:
    #     config.model.use_valid_num = int(args.use_valid_num)
    # print("set model.use_valid_num", config.model.use_valid_num)
    #
    # config.model.batchsize=args.batchsize
    # print("set model.batchsize", config.model.batchsize)
    # config.model.epoches = args.epoches
    # print("set model.epoches", config.model.epoches)
    # config.model.dropout_rate = args.dropout_rate
    # print("set model.dropout_rate", config.model.dropout_rate)
    # config.model.spa_drop_rate = args.spa_drop_rate
    # print("set model.spa_drop_rate", config.model.spa_drop_rate)
    # config.model.learning_rate = args.learning_rate
    # print("set model.learning_rate", config.model.learning_rate)
    #
    # config.model.use_liner_delay = args.use_liner_delay
    # print("set model.use_liner_delay", config.model.use_liner_delay)
    # config.model.use_liner_delay_epoch = args.use_liner_delay_epoch
    # print("set model.use_liner_delay_epoch", config.model.use_liner_delay_epoch)
    # config.model.learning_rate_ld_min = args.learning_rate_ld_min
    # print("set model.learning_rate_ld_min", config.model.learning_rate_ld_min)
    # config.model.learning_rate_ld_decay = args.learning_rate_ld_decay
    # print("set model.learning_rate_ld_decay", config.model.learning_rate_ld_decay)
    #
    # config.model.use_class_weight = args.use_class_weight
    # print("set use_class_weight", config.model.use_class_weight)
    # config.model.use_l1l2 = args.use_l1l2
    # print("set use_l1l2", config.model.use_l1l2)
    #
    # config.model.verbose = args.verbose
    # print("set verbose", config.model.verbose)
    #
    # config.save_model_name = args.save_model_name #+ time.ctime().replace(' ', '-')
    # print("set save_model_name", config.save_model_name)
    #
    # print(args.print_info)
       

                
    
    
