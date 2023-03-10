import pandas as pd
import numpy as np
# 导入数据：分隔符为空格
raw_data = pd.read_csv('C:/Users/76184/Desktop/data/cora/cora.content',sep = '\t',header = None)
num = raw_data.shape[0] # 样本点数2708
# 将论文的编号转[0,2707]
a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b,a)
map = dict(c)

# 将词向量提取为特征,第二行到倒数第二行
features =raw_data.iloc[:,1:-1]
 # 检查特征：共1433个特征，2708个样本点
print(features.shape) 
labels = pd.get_dummies(raw_data[1434])#提取标签进行独热编码
print(labels.head(3))#打印标签的前三行
#导入论文引用数据
raw_data_cites = pd.read_csv('C:/Users/76184/Desktop/data/cora/cora.cites',sep = '\t',header = None)
# 创建一个规模和邻接矩阵一样大小的矩阵
matrix = np.zeros((num,num))
# 创建邻接矩阵
for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
    x = map[i] ; y = map[j]  #替换论文编号为[0,2707]
    matrix[x][y] = matrix[y][x] = 1 #有引用关系的样本点之间取1
# 查看邻接矩阵的元素和（按每列汇总）
print(sum(matrix))


#-------------保存处理后的数据------------
np.save('cora_features.npy', features)
np.save('cora_labels.npy', labels)
np.save('cora_adj_matrix.npy', matrix)


