import torch
import random
import numpy as np
import pandas as pd
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, accuracy_score, f1_score
from torch_geometric.data import Data
from sklearn.model_selection import KFold
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, accuracy_score, f1_score, confusion_matrix, matthews_corrcoef)
from torch_geometric.utils import negative_sampling
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from FeatureProcessing import FeatureConstruction

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_metrics(y_true, y_pred_proba):
    y_pred = np.round(y_pred_proba)

    # 计算各种评估指标
    auc = roc_auc_score(y_true, y_pred_proba)
    aupr = average_precision_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred)  # 也称为灵敏度或真正率
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)  # 真负率或特异性
    mcc = matthews_corrcoef(y_true, y_pred)  # 马修斯相关系数

    return accuracy, recall, precision, specificity, f1, mcc

def GraphConstruction(ComplexFeatureVector, NodeFeatures, Labels, threshold=0.5):
    G = nx.Graph()
    edge_index = []
    all_rnas = list(NodeFeatures.keys())
    rna_to_idx = {rna: i for i, rna in enumerate(all_rnas)}
    # **构建节点特征矩阵 (3458, feature_dim)**
    feature_matrix = []
    for rna in all_rnas:
        # 确保 NodeFeatures[rna] 是一个由浮动数字构成的列表
        feature_str = NodeFeatures[rna]
        feature_vector = list(map(float, feature_str.split()))  # 将空格分隔的字符串转换为浮动数字
        feature_matrix.append(feature_vector)

    feature_matrix = np.array(feature_matrix, dtype=float)
    node_features_tensor = torch.tensor(feature_matrix, dtype=torch.float)  # (3458, feature_dim)

    # **计算相似性**
    similarity_matrix = cosine_similarity(feature_matrix)

    # **构造边**
    for i in range(len(all_rnas)):
        for j in range(i + 1, len(all_rnas)):  # 只遍历上三角
            similarity = similarity_matrix[i, j]
            if similarity > threshold:  # 设定阈值
                G.add_edge(all_rnas[i], all_rnas[j], weight=similarity)
                edge_index.append([rna_to_idx[all_rnas[i]], rna_to_idx[all_rnas[j]]])

    # **生成邻接矩阵**
    adj_matrix = nx.to_numpy_array(G, nodelist=all_rnas, weight="weight")

    # 生成标签张量
    labels = [Labels[rna] for rna in all_rnas]  # 收集所有RNA的标签
    # **确保所有标签是整数**

    labels = [int(label) for label in labels]  # 转换成整数
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return (
        torch.tensor(edge_index, dtype=torch.long).T,  # (2, num_edges)
        node_features_tensor,  # **节点特征，(3458, feature_dim)**
        torch.tensor(adj_matrix, dtype=torch.float),  # (3458, 3458)
        label_tensor  # **节点标签 (y)**
    )

def DataLoad(tv_set_path, test_set_path):
    list_tv_set = open(tv_set_path, 'r').readlines()
    list_test_set = open(test_set_path, 'r').readlines() if test_set_path != "None" else None
    return list_tv_set, list_test_set

def get_data():

    interaction = pd.read_csv("D:/Users/huawei/Desktop/adj_matrix.csv", header=None,
                              index_col=None)
    print('interaction',interaction.shape)
    interaction.columns = interaction.iloc[0]
    interaction = interaction.drop(0).reset_index(drop=True)
    # 读取药物相似度矩阵
    data = []
    with open("D:/Users/huawei/Desktop/NodeFeaturesj.txt", "r") as file:
        for line in file:
            values = line.split()[1:]  # 跳过第一列字符串
            data.append([float(x) for x in values])  # 转换为浮点数

    d_feature = np.array(data)
    # 读取蛋白质相似度矩阵
    data1 = []
    with open("D:/Users/huawei/Desktop/NodeFeaturesi.txt", "r") as file1:
        for line in file1:
            values = line.split()[1:]  # 跳过第一列字符串
            data1.append([float(x) for x in values])  # 转换为浮点数

    m_feature = np.array(data1)

    # 将相似度矩阵转换为 DataFrame
    d_feature = pd.DataFrame(d_feature)
    m_feature = pd.DataFrame(m_feature)

    m_emb = torch.FloatTensor(d_feature.values)
    print(m_emb.size())
    s_emb = torch.FloatTensor(m_feature.values)
    print(s_emb.size())
    m_emb = torch.cat([m_emb, torch.zeros(m_emb.size(0), max(m_emb.size(1), s_emb.size(1)) - m_emb.size(1))], dim=1)
    s_emb = torch.cat([s_emb, torch.zeros(s_emb.size(0), max(m_emb.size(1), s_emb.size(1)) - s_emb.size(1))], dim=1)

    feature = torch.cat([m_emb, s_emb]).cuda()

    l, p = interaction.values.nonzero()
    adj = torch.LongTensor([p, l + len(m_feature)]).cuda()
    data = Data(x=feature, edge_index=adj).cuda()
    print(data)
    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=True)(data)

    return splits


if __name__ == '__main__':
    data = get_data(2, 2048)
