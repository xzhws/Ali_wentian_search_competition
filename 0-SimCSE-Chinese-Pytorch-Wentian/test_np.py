from scipy.stats import spearmanr
import numpy as np
import torch

# y_true = torch.arange(10)
# use_row = torch.where((y_true + 1) % 3 != 0)[0]
#
# print(use_row)
# # tensor([0, 1, 3, 4, 6, 7, 9])
#
# y_true = (use_row - use_row % 3 * 2) + 1
#
# print(y_true) # 为什么真实标记是这样的呢
# # tensor([ 1,  0,  4,  3,  7,  6, 10])

# path = '/home/zqxie/project/SimCSE-Chinese-Pytorch-main/datasets/tianchi_data/tianchi_data_valid_neg.txt'
# path2 = ''
# with open(path, 'r', encoding='utf8') as f:
#     ans=  [(line.split("{}")[1], line.split("{}")[2], line.split("{}")[3]) for line in f]
#     print('len ans', len(ans))

from scipy.sparse import csr_matrix,csc_matrix
import numpy as np
indptr = np.array([0,2,3,6])
indices = np.array([0,2,2,0,1,2])
data = np.array([0.1,0.2,0.3,0.4,0.5,0.6])
csr_matrix_0 = csr_matrix((data,indices,indptr),shape=(3,3))
print(csr_matrix_0.nonzero())
print(csr_matrix_0.toarray())

