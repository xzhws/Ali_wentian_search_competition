import os.path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import random
from os.path import join
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import text_segmentate, truncate_sequences
import jieba

from sklearn.preprocessing import normalize
import logging
import sys

from keras.utils import multi_gpu_model
import json

logging.basicConfig(level=logging.INFO)
jieba.initialize()

# 基本信息 需要改动
epochs = 50
save_dir = "./output/res_finetune_unsupervised_100w_random_cut/"
model_dir = "./1-chinese_roformer-sim-char_L-12_H-768_A-12/chinese_roformer-sim-char_L-12_H-768_A-12/"
model_type = "roformer"
# 不需要改动的变量
maxlen = 70
batch_size =  80 # 调整
num_dim = 128
seq2seq_loss_ratio = 0.5
train_data_path = "./model_data/hold_out_t2s/train.txt" # 训练集 que-doc
dev_data_path = "./model_data/hold_out_t2s/dev.txt" # 验证集que-doc
# corpus_path = "./model_data/hold_out_t2s/dev.txt"

corpus_path = "model_data/hold_out_t2s/mini_corpus_clean.txt"
full_corpus_path = "model_data/hold_out_t2s/corpus_clean.txt"
mcpr_label_data_path = "/home/zqxie/project/SimCSE-Chinese-Pytorch-main/data/unsupervised_data_random_cut.txt"
# bert配置
config_path = join(model_dir, 'bert_config.json')
checkpoint_path = join(model_dir, 'bert_model.ckpt')
dict_path = join(model_dir, 'vocab.txt')
steps_per_epoch = 1000000 // batch_size

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

train_data_que = '/home/zqxie/project/SimCSE-Chinese-Pytorch-main/datasets/tianchi_data/train.query.txt'
doc_data = '/home/zqxie/project/SimCSE-Chinese-Pytorch-main/datasets/tianchi_data/corpus.tsv'
with open('./data/corpus.json','w') as wr:
    with open(doc_data,'r') as up:
        lines = up.readlines()
        for i, line in enumerate(lines):
            # line = lines[500]
            train_que_json = {}
            line = line.split('\t')[1]
            text_ids = tokenizer.encode(line)[0][1:-1] # 看一下后面输入bert的是否包含[cls] 和 [sep]两部分
            # print(line, text_ids)
            train_que_json['qid'] = str(i+1)
            train_que_json['input_ids'] = text_ids
            json.dump(train_que_json,wr)
            wr.write('\n')
    wr.close()
