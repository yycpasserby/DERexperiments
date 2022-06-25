# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:56:30 2019

@author: Administrator
"""
import pandas as pd
import gensim
import numpy as np
import myProgressbar as pb
from collections import Counter
import json
import os
import re

#try:
    #from keras import preprocessing
#except:
    #from tensorflow.keras import preprocessing

class ConfigWord(object):
    embedding_size = 300
    word2vec_source = './data/text81.model'
    sentence_len = 256
    sentence_avg_len = 10


class Model(object):
    batchsize = 32
    hidden_size = 256
    hidden_size_att = 256
    use_test_num = 0
    use_valid_num= 0
    use_liner_delay = 0
    use_tri_delay = 0        
    use_tri_delay_epoch = 20
    use_class_weight = 0
    use_att_reg = 0
    kernel_size = 3   
    use_l1l2 = None
    use_early_stop = 0.0
    mask_zero = False
    early_stop_step = 10
    early_stop_delay = 0.8
    early_stop_reset = 10
    early_stop_start_flag = False
    early_stop_reset_flag = False
    print_threshold = 0.95
    add_use_weighted = False
    metrics_average_tpye = 'macro'
    local_return_probabilities = False
    global_return_probabilities = False
    use_bias = True
    verbose = 1
    
    multi_channel_num = 3
    global_att_num = 3
    model_name = 'default'   
    epoches = 50
    dropout_rate = 0.2
    spa_drop_rate = 0.1
    learning_rate = 0.00001
    learning_rate_ld_min = 0.0005
    learning_rate_td_max = 0.0005
    learning_rate_td_min = 1e-6
    learning_rate_ld_decay = 0.98
    learning_rate_td_decay = 0.98
    learning_rate_td_frequency = 100
    class_weights= None
    pass

class Config(object):
    def __init__(self):
        
        if self.init is True:
            return
        else:
            self.init = True
        
        self.word = ConfigWord()
        self.model = Model()
        self.gpu_name='0'

        self.save_model_name = ''

        self.trainset_path = r'./data/meld_train5.csv'
        self.validset_path = r'./data/meld_dev5.csv'
        self.testset_path = r'./data/meld_test5.csv'
        self.vocab_path = './model/vocab'

        self.label_len = 224 #20 若输入向量，向量长*标签数
        self.use_tokenizer = False
#            

        self.stop_word_path = None
        
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Config, cls).__new__(cls)
            cls.init = False
        else:
            cls.init = True
        return cls.instance
    
    
class Dataset(object):
    '''
    '''
    def __init__(self, config):
        self._trainset_path = config.trainset_path
        self._validset_path = config.validset_path
        self._testset_path = config.testset_path
        self._stop_word_path = config.stop_word_path
        self._word_embedding_size = config.word.embedding_size
        self._word2vec_source = config.word.word2vec_source
        self._sentence_len = config.word.sentence_len

        self.train_content_word, self.train_label = None, None        
        self.valid_content_word, self.valid_label = None, None        
        self.test_content_word, self.test_label = None, None        


        self.vocab_path = config.vocab_path
                
        self.word_embedding = None
        
        self.oow_list = []
        self.stopWordDict=[]

        self._stop_word_path = ''#'./jieb/stopword.txt'

        self.use_tokenizer = bool(config.use_tokenizer)
        self.verbose = config.model.verbose
    
    def get_stop_word_dict(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(r'./data/meld_train1.csv', encoding='utf-8') as fp:
            for line in fp.readlines():
                self.stopWordDict.append(line.strip('\n'))

    def _read_data(self, filepath):

        data = pd.read_csv(filepath, encoding='utf-8',error_bad_lines=False, sep='\t')
        data_content = np.asarray(data.iloc[:,0])
        data_labels = np.array(data.iloc[:,1])
#        labels = pd.get_dummies(data_labels)[list(np.unique(data_labels))].values
        labels = data_labels
        
        widget = ['\r read_data: %s  '%filepath[0], pb.NumPercentage(), ' ', pb.Bar('#'), ' ', 
                  pb.Percentage(), ' ', pb.Timer(), ' ', pb.ETA(), ' ', 
                  pb.FileTransferSpeed('line')] 
        bar = pb.MyProgressBar(maxval=len(data_content), widgets=widget)
        if self.verbose != 0:
            bar.start()
        word_contents = []
#        labels = []
        for i in range(0, len(data_content)): 
            jieba_res = data_content[i].split()
            new_jieba_res = []
            for word in jieba_res:
                if word in self.stopWordDict:
                    continue
                else:
                    new_jieba_res.append(word);
            word_contents.append(new_jieba_res)

            if self.verbose != 0:
                bar.update(i + 1)
                
        if self.verbose != 0:
            bar.finish()
        
        contents = word_contents
        
        return contents, labels
    
    #def _is_chinese(slef, word):
    #    for ch in word:
    #        if '\u4e00' <= ch <= '\u9fff':
    #            return True
    #        else:
    #            return False
        
    def _get_word_embedding(self, words):
        '''
        可使用tokenizer 
        '''
        word2vec = {}
        with open(r'./data/glove.6B.300d.txt', encoding='utf-8') as fp:
            for line in fp.readlines():
                lineList = line.split()
                word = lineList[0]
                vector = np.array(lineList[1:])
                word2vec[word] = vector

        # word_vec = gensim.models.Word2Vec.load(self._word2vec_source)
        # word_vec = {}
        word_vec = word2vec
        vocab_word = []
        word_embedding = []
        
        # 添加 "pad" 和 "UNK", 
        vocab_word.append("pad")
        vocab_word.append("UNK")
        vocab_word.append("<eos>")
        vocab_word.append("__eou__")
        word_embedding.append(np.zeros(self._word_embedding_size))
        word_embedding.append(np.random.randn(self._word_embedding_size))
        word_embedding.append(np.ones(self._word_embedding_size))
        word_embedding.append(np.ones(self._word_embedding_size))
        
        for word in words:
            try:
                vector = word_vec[word]
                vocab_word.append(word)
                word_embedding.append(vector)
            except:
                print(word + "不存在于词向量中")
                self.oow_list.append(word)

        return vocab_word, np.array(word_embedding)

    def _rebuild_process(self, contents, sequence_length, some_to_index):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """
        contents_vec = np.zeros((sequence_length))
        sequence_len = sequence_length
        
        # 判断当前的序列是否小于定义的固定序列长度
        if len(contents) < sequence_length:
            sequence_len = len(contents)
            
        for i in range(sequence_len):
            if contents[i] in some_to_index:
                contents_vec[i] = some_to_index[contents[i]]
            else:
                contents_vec[i] = some_to_index["UNK"]
    
        return contents_vec
    
    def _gen_train_eval_data(self, x_train_word_content, y_train_word_label, info='test'):
        """
        生成训练集和测试集
        """
        
        train_word_contents = []
        train_labels = []
        
        # 遍历所有的文本，将文本中的词转换成index表示
        if self.use_tokenizer is False:
            widget = ['\r make %s  :'%info, pb.NumPercentage(), ' ', pb.Bar('#'), ' ', 
                      pb.Percentage(), ' ', pb.Timer(), ' ', pb.ETA(), ' ', 
                      pb.FileTransferSpeed('line')] 
            if self.verbose != 0:
                bar = pb.MyProgressBar(maxval=len(x_train_word_content), widgets=widget).start()
    
            for i in range(len(x_train_word_content)):
                content_vec = self._rebuild_process(x_train_word_content[i], 
                                                    self._sentence_len, self._word_to_index)
                train_word_contents.append(content_vec)
                train_labels.append(y_train_word_label[i])
                if self.verbose != 0:
                    bar.update(i)
            
            if self.verbose != 0:
                bar.finish()
        else:
            sequences = self.word_tokenizer.texts_to_sequences(x_train_word_content)
            train_word_contents = preprocessing.sequence.pad_sequences(sequences, maxlen=self._sentence_len, padding='post')

            train_labels = y_train_word_label
        train_word_contents = np.asarray(train_word_contents, dtype="int64")
        train_labels = np.array(train_labels)
       
        
        return train_word_contents, train_labels
    
    def _gen_vocabulary(self, train_contents_word, eval_contents_word, test_contents_word):
        """
        生成词向量和词汇-索引映射字典
        """
        
        all_words = []
        if train_contents_word is not None:
            all_words += [word for content in train_contents_word for word in content]
        if eval_contents_word is not None:
            all_words += [word for content in eval_contents_word for word in content]
        if test_contents_word is not None:
            all_words += [word for content in test_contents_word for word in content]


        if self.use_tokenizer is True:
            self.data_process_tokenizer(all_words)
        else:
        # 去掉停用词
#            subWords=[]
#            for word in all_words:
#                if self._is_chinese(word) is False or word in self.stopWordDict:
#                    continue
#                else:
#                    subWords.append(word)
#
    #        subWords = [word for word in allWords if _is_chinese(word) is False and word not in self.stopWordDict else word]

            word_count = Counter(all_words)  # 统计词频
            sort_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

            # 去除低频词
            words = [item[0] for item in sort_word_count if item[1] >= 5]

            # if self.vocab_path is not None:
            #     with open(self.vocab_path, 'w', encoding='utf-8') as fp:
            #         for word in words:
            #             fp.write("%s\n"%word)
        # elif os.path.exists(self.vocab_path):
        #     words = [line.strip() for line in open(self.vocab_path, 'r', encoding='utf-8').readlines()]
        # else:
        #     print('input vocab path')
        
        print("all word:", len(words), words[:10])
        vocab_word, word_embedding = self._get_word_embedding(words)
        self.word_embedding = word_embedding
        
        self._word_to_index = dict(zip(vocab_word, list(range(len(vocab_word)))))

    def data_process_tokenizer(self, words):
        voc_size = len(set(words))
        
        tokenizer = preprocessing.text.Tokenizer(nb_words=None, lower=False)

        tokenizer.fit_on_texts(words)
        print(len(tokenizer.word_index))
        embedding_matrix = gensim.models.Word2Vec.load(self._word2vec_source)

        embed_train_matrix = np.zeros((voc_size+1,self._word_embedding_size))
        unk_vector = np.random.randn(self._word_embedding_size)
        for w,i in tokenizer.word_index.items():
            try:
                embedding_vector=embedding_matrix[w]
                embed_train_matrix[i] = embedding_vector
            except:
                embed_train_matrix[i] = unk_vector
        self.word_embedding  =  embed_train_matrix
        self.word_tokenizer = tokenizer

        #Glove词嵌入
        # glove_dir = r'F:\数据集\glove.6B'
        # embeddings_index = {}
        # f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), 'r', encoding='UTF-8')
        # for line in f:
        #     values = line.split()
        #     word = values[0]
        #     coefs = np.asarray(values[1:], dtype='float32')
        #     embeddings_index[word] = coefs
        # f.close()
        # print('Found %s word vectors.' % len(embeddings_index))
        # embedding_dim = 100
        # embedding_matrix = np.zeros((max_words, embedding_dim))
        # for word, i in word_index.items():
        #     if i < max_words:
        #         embedding_vector = embeddings_index.get(word)
        #         if embedding_vector is not None:
        #             # Words not found in embedding index will be all-zeros.
        #             embedding_matrix[i] = embedding_vector

      
    def data_gen(self):
        
        self.get_stop_word_dict(self._stop_word_path)
        if self._trainset_path is not None and os.path.exists(self._trainset_path):
            train_content, train_label = self._read_data(self._trainset_path)
        else:
            train_content, train_label = None, None
    
        if self._validset_path is not None and os.path.exists(self._validset_path):
            valid_content, valid_label = self._read_data(self._validset_path)
        else:
            valid_content, valid_label = None, None

        if self._testset_path is not None and os.path.exists(self._testset_path):
            test_content, test_label = self._read_data(self._testset_path)
        else:
            test_content, test_label = None, None


        self._gen_vocabulary(train_content, valid_content, test_content)
            
        # 初始化训练集和测试集
        if train_content is not None:
            self.train_content_word, self.train_label = self._gen_train_eval_data(train_content, train_label)
            self.train_token = train_content
        if valid_content is not None:
            self.valid_content_word, self.valid_label = self._gen_train_eval_data(valid_content, valid_label)
            self.valid_token = valid_content
        if test_content is not None:
            self.test_content_word, self.test_label = self._gen_train_eval_data(test_content, test_label)
            self.test_token = test_content


def next_batch(content, label, batch_size):
        """
        生成batch数据集，用生成器的方式输出
        """
        lenth = len(label)
        perm = np.arange(lenth)
        np.random.shuffle(perm)
        x = content[0][perm]
        y = content[1][perm]
        z = content[2][perm]
        
        l = label[perm]
        
        for start in range(0, lenth, batch_size):
            end = min(start + batch_size, lenth)
            batchX = np.array(x[start: end], dtype="int64")
            batchY = np.array(y[start: end], dtype="int64")
            batchZ = np.array(z[start: end], dtype="float32")
            batchL = np.array(l[start: end], dtype="float32")
            print('index: ', start, '\r\n')
            yield [batchX, batchY, batchZ], batchL

