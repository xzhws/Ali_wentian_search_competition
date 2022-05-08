# -*- encoding: utf-8 -*-

import random
import time
from typing import List

import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
from sklearn.preprocessing import normalize
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 基本参数
EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-5
MAXLEN = 64
POOLING = 'cls'  # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预训练模型目录
BERT = 'pretrained_model/bert_pytorch'
BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
ROBERTA = './pretrained_model/chinese_roberta_wwm_ext_pytorch'
model_path = ROBERTA
# saved_cse_path = './saved_model/simcse_sup.pt'
# model_path = BERT

# 微调后参数存放位置
SAVE_PATH = './saved_model/simcse_sup_data_valid_NEZHA.pt'
save_path_dir = SAVE_PATH[:-3]+ '/'
os.makedirs(save_path_dir, exist_ok=True)

# 数据位置
SNIL_TRAIN = './datasets/cnsd-snli/train.txt'
STS_DEV = './datasets/STS-B/cnsd-sts-dev.txt'
STS_TEST = './datasets/STS-B/cnsd-sts-test.txt'

TIANCHI_TRAIN = './datasets/tianchi_data/tianchi_data_train.txt'
TIANCHI_VALID = './datasets/tianchi_data/tianchi_data_valid.txt'
TIANCHI_VALID_NEG = './datasets/tianchi_data/tianchi_data_valid_neg.txt'
# tianchi_data_valid_neg.txt
TIANCHI_TEST = './datasets/tianchi_data/dev.query.txt'
TIANCHI_TEST_CORPUS = './datasets/tianchi_data/corpus.tsv'
# TIANCHI_TRAIN_QUERY = './datasets/tianchi_data/tra'
tokenizer = BertTokenizer.from_pretrained(model_path)

def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集"""

    def load_tianchi_test(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t') for line in f]

    def load_tianchi_valid(path):
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("{}")[1], line.split("{}")[2], line.split("{}")[3]) for line in f]

    assert name in ["snli", "lqcmc", "sts", 'tianchi', 'tianchi_test', 'tianchi_valid']

    if name == 'tianchi_test':
        return load_tianchi_test(path)
    if name == 'tianchi_valid':
        return load_tianchi_valid(path)
    else:
        return None

class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True,
                         padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        line = self.data[index]
        return self.text_2_id([line[0]]), self.text_2_id([line[1]]), int(line[2])  # (1,2) 输入数据 (3)表示这组数据相关的标签



class SimcseModel(nn.Module):
    """Simcse有监督模型定义"""

    def __init__(self, pretrained_model: str, pooling: str):
        super(SimcseModel, self).__init__()
        # config = BertConfig.from_pretrained(pretrained_model)   # 有监督不需要修改dropout
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.pooling = pooling
        # 映射成128维
        self.dense_model = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):

        # out = self.bert(input_ids, attention_mask, token_type_ids)
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            out = out.last_hidden_state[:, 0]  # [batch, 768]

        elif self.pooling == 'pooler':
            out = out.pooler_output  # [batch, 768]

        elif self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            out = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        elif self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            out = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        out = self.dense_model(out)
        return out

def eval_doc(model, dataloader) -> float:
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    with torch.no_grad():
        for source, target, label in tqdm(dataloader):
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source['input_ids'].squeeze(1).to(DEVICE)
            source_attention_mask = source['attention_mask'].squeeze(1).to(DEVICE)
            source_token_type_ids = source['token_type_ids'].squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)  #

            target_input_ids = target['input_ids'].squeeze(1).to(DEVICE)
            target_attention_mask = target['attention_mask'].squeeze(1).to(DEVICE)
            target_token_type_ids = target['token_type_ids'].squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)
            # print('sim:{}'.format(sim))
            # break
        sim_tensor = sim_tensor.detach().cpu().numpy()

        arg_max = sim_tensor.argsort() # 最低相似度的例子
        with open('{}/{}.txt'.format(save_path_dir, 'badcases'), 'w') as up:
            for item in arg_max[:5000]: # 最差的5000条样本
                # print('que:{}\tdoc:{}\tsim_score:{:.4f}\n'.format(dev_data[item][0], dev_data[item][1], sim_tensor[item]))
                up.write('{}\t{}\t{}\t{:.4f}\n'.format(item, dev_data[item][0], dev_data[item][1],sim_tensor[item]))
            up.close()

def get_embedding():
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    ### 加载测试集和商品库
    key = TIANCHI_TRAIN  # TIANCHI_TEST_CORPUS, TIANCHI_TEST
    global dev_data
    dev_data = load_data('tianchi_valid', key)

    dev_dataloader = DataLoader(TestDataset(dev_data), batch_size=BATCH_SIZE)
    print(dev_data[0], '\n', dev_data[-1])

    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH))

    eval_doc(model, dev_dataloader)
if __name__ == '__main__':
    get_embedding()

