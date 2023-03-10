from sklearn.model_selection import train_test_split
import os
import numpy as np
from torch.utils.data import  DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GCN_model import GCN
#-------------------导入数据
features = np.load('cora_features.npy')
labels = np.load('cora_labels.npy')
adj_matrix = np.load('cora_adj_matrix.npy')

features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, stratify=labels)
features_val, features_test, labels_val, labels_test = train_test_split(features_test, labels_test, test_size=0.5, stratify=labels_test)


model= GCN(num_features=features.shape[1],hidden_size=16,num_classes=labels.unique().size()[0])
#----优化器------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#------------损失函数--------
criterion = nn.CrossEntropyLoss()
#定义评估指标：准确率
def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels)
    acc = correct.float() / len(labels)
    return acc
                
