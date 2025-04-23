import torch
import numpy as np
import argparse
from mask import *
from utils import get_data, set_seed, prepare_cross_validation_data
from model import GNNEncoder, EdgeDecoder,  DegreeDecoder, Encoder, GMAE, GAE, GraphSAGE,GINEncoder,GATEncoder,GCNEncoder,SAGEEncoder,LPDecoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# main parameter
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023, help="Random seed for model and dataset.")
parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
parser.add_argument('--p', type=float, default=0.6, help='Mask ratio')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_layers', type=int, default=2)
args = parser.parse_args()
set_seed(args.seed)

# {'AUC': '0.985314', 'AP': '0.980734', 'ACC': '0.949944', 'SEN': '0.975528', 'PRE': '0.928042', 'SPE': '0.924360', 'F1': '0.951193', 'MCC': '0.901069'}

# 初始化保存所有运行的最佳指标的列表
all_runs_metrics = []
# 初始化累积指标值的字典
cumulative_metrics = {'AUC': [], 'AP': [], 'ACC': [], 'SEN': [], 'PRE': [], 'SPE': [], 'F1': [], 'MCC': []}

encoder = Encoder(in_channels=194, hidden_channels=64, out_channels=128)
edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
lp_decoder = LPDecoder(in_channels=128, hidden_channels=64, out_channels=1, encoder_layer=1, num_layers=args.num_layers,
                     dropout=args.dropout)
degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
mask = MaskPath(p=args.p)
best_metrics = {}
best_auc = 0.0

plt.figure(figsize=(10, 8))  # For ROC curves

# Additional figure for PR curves
plt.figure(figsize=(10, 8))  # For PR curves

# Initialize variables for ROC and PR curve plotting
all_fpr = np.linspace(0, 1, 100)
tprs = []
mean_tpr = 0.0
aucs = []
precisions = []  # For PR curves
mean_precision = 0.0  # For PR curves
average_precision_scores = []  # For PR curves

# 存储每一折的AUC和AUPRC
fold_aucs = []
fold_auprcs = []
# 用于存储所有折的真阳性率（TPR）值
mean_tpr = np.zeros_like(all_fpr)
for run in range(10):
    set_seed(args.seed + run)

    model = GMAE(encoder, lp_decoder, degree_decoder, mask).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits= get_data()
    # 每次运行的累计指标初始化
    run_metrics = {'AUC': 0, 'AP': 0, 'ACC': 0, 'SEN': 0, 'PRE': 0, 'SPE': 0, 'F1': 0, 'MCC': 0}
    num_epochs = 0
    y_true_all, y_scores_all = [], []
    for epoch in range(1000):
        model.train()
        train_data = splits['train']
        x, edge_index = train_data.x, train_data.edge_index
        loss = model.train_epoch(splits['train'], optimizer, alpha=args.alpha)
        model.eval()
        test_data = splits['test']
        z = model.encoder(test_data.x, test_data.edge_index)

        test_auc, test_aupr, acc, pre, sen, F1, mcc, y_true, y_scores = model.test(z, test_data.pos_edge_label_index,
                                                                                   test_data.neg_edge_label_index)

        # 根据新的返回值更新results字典，包括SEN和MCC
        results = {
            'AUC': "{:.6f}".format(test_auc),
            'AUPR': "{:.6f}".format(test_aupr),
            "Acc": "{:.6f}".format(acc),
            "Pre": "{:.6f}".format(pre),
            "SEN": "{:.6f}".format(sen),  # 使用SEN代替Recall
            "F1": "{:.6f}".format(F1),
            "MCC": "{:.6f}".format(mcc),  # 新增MCC
        }
        print(results)
        # 累加这些指标到run_metrics中
        run_metrics['AUC'] += test_auc
        run_metrics['AP'] += test_aupr
        run_metrics['ACC'] += acc
        run_metrics['SEN'] += sen
        run_metrics['PRE'] += pre
        run_metrics['F1'] += F1
        run_metrics['MCC'] += mcc

        num_epochs += 1  # 更新迭代次数
 # 计算这次运行的指标平均值并保存
    run_metrics = {metric: total / num_epochs for metric, total in run_metrics.items()}
    all_runs_metrics.append(run_metrics)
    # 累积每次运行的指标值
    for metric in cumulative_metrics.keys():
        cumulative_metrics[metric].append(run_metrics[metric])

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    tprs.append(np.interp(all_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # Calculate PR curve and Average Precision (AP)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    average_precision_scores.append(average_precision)

    # Interpolate precision to get smooth PR curve
    recall_levels = np.linspace(0, 1, 100)
    interpolated_precision = np.interp(recall_levels, recall[::-1], precision[::-1])
    interpolated_precision[0] = 1.0  # Precision starts at 1 at 0 recall
    precisions.append(interpolated_precision)

    # Plot ROC for each run
    plt.figure(1)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {run + 1} (AUC = {roc_auc:.4f})')

    # Plot PR for each run
    plt.figure(2)
    plt.plot(recall, precision, lw=1, alpha=0.3, label=f'Fold {run + 1} (AP = {average_precision:.4f})')


# Compute mean ROC
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(all_fpr, mean_tpr)
std_auc = np.std(aucs)

# Compute mean PR
mean_precision = np.mean(precisions, axis=0)
mean_ap = np.mean(average_precision_scores)
std_ap = np.std(average_precision_scores)

# Plot mean ROC
plt.figure(1)
plt.plot(all_fpr, mean_tpr, color='r', label=f'Mean ROC (AUC = {mean_auc:.4f} $\pm$ {std_auc:.4f})', lw=2, alpha=0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=16)
plt.ylabel('True Positive Rate',fontsize=16)
plt.title('Receiver Operating Characteristic',fontsize=20)
plt.legend(loc="lower right", fontsize=16)
plt.xlim([-0.05, 1.05])
plt.ylim([0.0, 1.05])

# Plot mean PR
plt.figure(2)
plt.plot(recall_levels, mean_precision, color='r', label=f'Mean PR (AP = {mean_ap:.4f} $\pm$ {std_ap:.4f})', lw=2, alpha=0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=16)
plt.ylabel('Precision',fontsize=16)
plt.title('Precision-Recall Curve',fontsize=20)
plt.legend(loc="lower left", fontsize=16)
plt.xlim([-0.05, 1.05])
plt.ylim([0.0, 1.05])
plt.tight_layout()


# Show both plots
plt.show()

# 打印每次运行的平均指标
for i, run_metrics in enumerate(all_runs_metrics, 1):
    print(f"Run {i} Metrics:")
    for metric, value in run_metrics.items():
        print(f"{metric}: {value:.6f}")

# 计算每个指标的平均值和标准差，并打印结果
for metric, values in cumulative_metrics.items():
    mean_value = np.mean(values)
    std_dev = np.std(values)
    print(f"{metric}: Mean = {mean_value:.6f}, Std Dev = {std_dev:.6f}")

#print(f"Mean AUC: {mean_auc:.6f} ± {std_auc:.6f}")

# # 打印每次运行的平均指标
# for i, auc_value in enumerate(run_auc_list, 1):
#     print(f"Run {i} AUC: {auc_value:.4f}")
# print("-" * 20)
#
# average_metrics = {metric: np.mean([run[metric] for run in all_runs_metrics]) for metric in all_runs_metrics[0]}
#
# # 打印十次运行的每次的平均指标
# for i, metrics in enumerate(all_runs_metrics, 1):
#     print(f"Run {i} Metrics:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value:.6f}")
#     print("-" * 20)
#
# # 打印十次运行的整体平均指标
# print("Overall Average Metrics after all runs:")
# for metric, value in average_metrics.items():
#     print(f"{metric}: {value:.6f}")



# # 比较和更新最高指标
#     if test_auc > best_auc:
#         best_auc = test_auc
#         best_metrics = results
#
# # 循环结束后打印最高 AUC 对应的指标
# print("Best Metrics with highest AUC:")
# for key, value in best_metrics.items():
#     try:
#         value = float(value)  # 尝试将值转换为浮点数
#         print(f"{key}: {value:.6f}")
#     except ValueError:
#         print(f"{key}: {value}")  # 如果转换失败，直接打印原始字符串
#


