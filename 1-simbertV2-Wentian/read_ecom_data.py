import numpy as np
from tqdm import tqdm

train_que = './ecom/train.query.txt'
doc = './ecom/corpus.tsv'
que_doc = './ecom/qrels.train.tsv'

que_doc_txt = './ecom/que_doc.txt'

que_list = []
doc_list = []

with open(train_que, 'r') as up:
    # cnt = 0
    for line in up.readlines():
        que_list.append(line.split('\t')[1][:-1])
        # cnt += 1
        # if cnt>50:
        #     break

with open(doc, 'r') as up:
    # cnt = 0
    for line in tqdm(up.readlines()):
        # print(line.split('\t'))
        doc_list.append(line.split('\t')[1][:-1])
        # cnt += 1
        # if cnt > 50:
        #     break

with open(que_doc_txt, 'w') as wr:
    with open(que_doc,'r') as up:
        # cnt = 0
        for line in up.readlines():
            line = line.split('\t')
            que_id, doc_id = int(line[0])-1, int(line[2])-1
            # print('que:{} doc:{}'.format(que_list[que_id], doc_list[doc_id]))
            wr.write('{}\t{}\n'.format(que_list[que_id], doc_list[doc_id]))
            # cnt += 1
            # if cnt > 100:
            #     break