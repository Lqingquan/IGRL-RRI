import numpy as np
import pandas as pd


def FeatureConstruction(ListPair):
    """
    解析 RNAi 和 RNAj，并存储它们的配对关系及标签
    """
    Labels = {}  # 存储 RNAi-RNAj 交互的标签
    RNA_pairs = set()  # 存储 RNAi 和 RNAj 的所有配对关系

    for LinePair in ListPair:
        RNAiname, RNAjname, RNAisequence, RNAjsequence, RNAistructure, RNAjstructure, label = LinePair.strip().split(
            ',')

        RNA_pairs.add((RNAiname, RNAjname))  # 记录 RNAi 和 RNAj 的配对
        Labels[(RNAiname, RNAjname)] = int(label)  # 记录它们的交互情况（1/0）

    return Labels, RNA_pairs


def construct_adjacency_matrix(RNA_pairs, Labels, output_file):
    """
    根据 RNAi 和 RNAj 构建邻接矩阵，并根据 Labels 标记交互关系
    """
    # 获取所有 RNA 的唯一名称
    RNAi_names = sorted(set(rna_i for rna_i, _ in RNA_pairs))
    RNAj_names = sorted(set(rna_j for _, rna_j in RNA_pairs))

    # 初始化邻接矩阵 (len(RNAj) 行, len(RNAi) 列)
    adj_matrix = np.zeros((len(RNAj_names), len(RNAi_names)), dtype=int)

    # 构建 RNA 名字到索引的映射
    rna_i_to_idx = {rna: i for i, rna in enumerate(RNAi_names)}
    rna_j_to_idx = {rna: j for j, rna in enumerate(RNAj_names)}

    # 填充邻接矩阵
    for (rna_i, rna_j), label in Labels.items():
        if rna_i in rna_i_to_idx and rna_j in rna_j_to_idx:
            adj_matrix[rna_j_to_idx[rna_j], rna_i_to_idx[rna_i]] = label

    # 保存邻接矩阵到 CSV
    df = pd.DataFrame(adj_matrix, index=RNAj_names, columns=RNAi_names)
    df.to_csv(output_file, index=True, header=True)

def DataLoad(tv_set_path, test_set_path):
    list_tv_set = open(tv_set_path, 'r').readlines()
    list_test_set = open(test_set_path, 'r').readlines() if test_set_path != "None" else None
    return list_tv_set, list_test_set
list_tv_set, list_test_set = DataLoad('D:/pycharm/RNAI-FRID-main/RNAI-FRID-main/Example/TrainingValidationSet.fasta', 'D:/pycharm/RNAI-FRID-main/RNAI-FRID-main/Example/TestSet.fasta')
# 解析 RNAi-RNAj 关系和交互标签
Labels, RNA_pairs = FeatureConstruction(list_test_set)

# 生成邻接矩阵
output_file = "D:/Users/huawei/Desktop/test_adj_matrix.csv"
construct_adjacency_matrix(RNA_pairs, Labels, output_file)