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
# 基本参数
EPOCHS = 2
BATCH_SIZE = 64
LR = 1e-5
MAXLEN = 64
POOLING = 'cls'  # choose in ['cls', 'pooler', 'last-avg', 'first-last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预训练模型目录
BERT = 'pretrained_model/bert_pytorch'
BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
NEZHA_WWM_EXT = 'pretrained_model/nezha-base-www'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
ROBERTA = './pretrained_model/chinese_roberta_wwm_ext_pytorch'
model_path = NEZHA_WWM_EXT
# model_path = BERT

# 微调后参数存放位置
SAVE_PATH = './saved_model/simcse_sup_data_valid_NEZHA.pt'

# 数据位置
SNIL_TRAIN = './datasets/cnsd-snli/train.txt'
STS_DEV = './datasets/STS-B/cnsd-sts-dev.txt'
STS_TEST = './datasets/STS-B/cnsd-sts-test.txt'

TIANCHI_TRAIN = './datasets/tianchi_data/tianchi_data_train.txt'
TIANCHI_VALID = './datasets/tianchi_data/tianchi_data_valid.txt'
TIANCHI_VALID_NEG = './datasets/tianchi_data/tianchi_data_valid_neg.txt'
# tianchi_data_valid_neg.txt
TIANCHI_TEST = './datasets/tianchi_data/dev.query.txt'


def load_data(name: str, path: str) -> List:
    """根据名字加载不同的数据集"""

    def load_snli_data(path):
        with jsonlines.open(path, 'r') as f:
            return [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    def load_lqcmc_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[0] for line in f]

    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    # assert name in ["snli", "lqcmc", "sts"]
    # if name == 'snli':
    #     return load_snli_data(path)
    # return load_lqcmc_data(path) if name == 'lqcmc' else load_sts_data(path)
    def load_tianchi_train_valid(path):
        # 不需要真实标签
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("{}")[1], line.split("{}")[2], line.split("{}")[3]) for line in f]

    def load_tianchi_test(path):
        with open(path, 'r', encoding='utf8') as f:
            return [line.strip().split('\t')[1] for line in f]

    assert name in ["snli", "lqcmc", "sts", 'tianchi', 'tianchi_test']
    if name == 'snli':
        return load_snli_data(path)
    elif name == 'lqcmc':
        return load_lqcmc_data(path)
    elif name == 'sts':
        return load_sts_data(path)
    elif name == 'tianchi':
        return load_tianchi_train_valid(path)
    elif name == 'tianchi_test':
        return load_tianchi_test(path)
    else:
        return None


class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        return tokenizer([text[0], text[1]], max_length=MAXLEN,
                         truncation=True, padding='max_length', return_tensors='pt')  # (1,2) 正样本， (1,3)负样本

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


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


def simcse_sup_loss(y_pred: 'tensor') -> 'tensor':
    """有监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 3, 768]

    """
    # 得到y_pred对应的label, 每第三句没有label, 跳过, label= [1, 0, 4, 3, ...]
    ''''没搞懂
    为什么这里的label要这么设计呢
    '''
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true) #

    return loss

# def eval_loss(model, dataloader) -> float: # 看验证集的loss来选择分类的样本正确率
def eval_loss_validation(model, dataloader):
    # 有监督的损失函数
    model.eval()
    loss_valid = 0
    cnt = 0
    with torch.no_grad():
        for batch_idx, source in enumerate(dataloader, start=1):
            # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
            # 训练
            y_pred = model(input_ids, attention_mask, token_type_ids)

            y_true = torch.arange(y_pred.shape[0], device=DEVICE)
            y_true = (y_true - y_true % 2 * 2) + 1
            # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
            sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
            # 将相似度矩阵对角线置为很小的值, 消除自身的影响
            sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
            # 相似度矩阵除以温度系数
            sim = sim / 0.05
            # 计算相似度矩阵与y_true的交叉熵损失
            loss = F.cross_entropy(sim, y_true)
            loss_valid += loss
            cnt += 1

    return loss_valid / cnt

def eval(model, dataloader) -> float:
    """模型评估函数
    批量预测, 计算cos_sim, 转成numpy数组拼接起来, 一次性求spearman相关度
    """
    model.eval()
    # sim_tensor = torch.tensor([], device=DEVICE)
    # label_array = np.array([])
    score = []
    with torch.no_grad():
        for batch_idx, source in enumerate(tqdm(train_dl), start=1):
            # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
            attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
            # 训练
            out = model(input_ids, attention_mask, token_type_ids)
            loss = simcse_sup_loss(out)

    return np.mean(np.array(score))


def train(model, train_dl, dev_dl, optimizer) -> None:
    """模型训练函数
    """
    model.train()
    global best
    early_stop_batch = 0
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 3, seq_len] -> [batch * 3, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
        # 训练
        out = model(input_ids, attention_mask, token_type_ids)
        # print('out shape:{}'.format(out.size())) # [batch_size*3, 128]
        loss = simcse_sup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 评估
        if batch_idx % 10 == 0:
            logger.info(f'loss: {loss.item():.4f}')
            # corrcoef = eval(model, dev_dl)
            loss_valid = eval_loss_validation(model, dev_dl)
            # torch.save(model.state_dict(), SAVE_PATH)
            model.train()
            if best > loss_valid: # 欧氏距离平均值, 损失函数
                early_stop_batch = 0
                best = loss_valid
                torch.save(model.state_dict(), SAVE_PATH)
                logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
                continue
            # early_stop_batch += 1
            # if early_stop_batch == 10:
            #     logger.info(f"corrcoef doesn't improve for {early_stop_batch} batch, early stop!")
            #     logger.info(f"train use sample number: {(batch_idx - 10) * BATCH_SIZE}")
            #     return


if __name__ == '__main__':
    # SAMPLES = 1000
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # load data
    # train_data = load_data('snli', SNIL_TRAIN)
    train_tianchi = load_data('tianchi', TIANCHI_TRAIN)
    # train_tianchi = random.sample(train_tianchi, SAMPLES)
    random.shuffle(train_tianchi)

    # dev_data = load_data('sts', STS_DEV)
    # test_data = load_data('sts', STS_TEST)
    dev_data = load_data('tianchi', TIANCHI_VALID)
    train_tianchi = train_tianchi + dev_data[:-1000]
    dev_data = dev_data[-1000:]
    # dev_dada_neg = load_data('tianchi', TIANCHI_VALID_NEG)
    # from random import sample
    # dev_dada_neg = sample(dev_dada_neg, len(dev_data))

    # print(len(dev_data), len(dev_dada_neg))
    # dev_data = dev_data + dev_dada_neg
    # dev_data = dev_data[:1000] + dev_dada_neg[:1000]
    # random.shuffle(dev_data)
    print(dev_data[0],'\n', dev_data[-1])
    # dev_data = random.sample(dev_data, SAMPLES)

    print('len(train):{} len(valid):{}'.format(len(train_tianchi), len(dev_data)))

    train_dataloader = DataLoader(TrainDataset(train_tianchi), batch_size=BATCH_SIZE)
    dev_dataloader = DataLoader(TrainDataset(dev_data), batch_size=BATCH_SIZE)
    # test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE)
    # load model
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # train
    best = 1e12 # 距离度量
    for epoch in range(EPOCHS):
        logger.info(f'epoch: {epoch}')
        train(model, train_dataloader, dev_dataloader, optimizer)
    logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    # eval
    # model.load_state_dict(torch.load(SAVE_PATH))
    # dev_corrcoef = eval(model, dev_dataloader)
    # logger.info(f'dev_corrcoef: {dev_corrcoef:.4f}')
