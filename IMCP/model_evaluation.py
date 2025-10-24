# model_evaluation.py
import numpy as np
import pandas as pd
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


def evaluate_model(model, X_test, y_test, save_path=None):
    # 模型预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    logloss = log_loss(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # 打印评估结果
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"特异性: {specificity:.4f}")
    print(f"Log损失: {logloss:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)

    # 可视化混淆矩阵
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Cancerous', 'Cancerous'], yticklabels=['Non-Cancerous', 'Cancerous'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if save_path:
        plt.savefig(f"{save_path}/confusion_matrix.png")
    plt.close()

    return accuracy, auc, conf_matrix

def delong_test(y_true, pred_1, pred_2):
    """
    比较两个模型AUC差异的DeLong检验
    返回: z_score, p_value
    """
    # 计算协方差
    n = len(y_true)
    var = np.var(pred_1 - pred_2)
    auc1 = roc_auc_score(y_true, pred_1)
    auc2 = roc_auc_score(y_true, pred_2)
    
    # 计算z值
    z = (auc1 - auc2) / np.sqrt(var/n)
    p = 2 * norm.sf(abs(z))  # 双侧检验
    
    return z, p

def compare_models_delong(models_results, model_names, save_path=None):
    print("\n=== DeLong检验结果 ===")
    comparisons = []
    # 遍历所有两两组合（i < j）
    for i in range(len(models_results)):
        for j in range(i+1, len(models_results)):
            model_i = models_results[i]
            model_j = models_results[j]
            y_true = model_i["target_data"]
            
            # 动态生成预测概率（避免缓存问题）
            X_i = model_i["features_data"]
            X_j = model_j["features_data"]
            prob_i = model_i["model"].predict_proba(X_i)[:, 1]
            prob_j = model_j["model"].predict_proba(X_j)[:, 1]
            
            # 执行检验
            z_score, p_value = delong_test(y_true, prob_i, prob_j)
            comparisons.append({
                "model1": model_names[i],
                "model2": model_names[j],
                "z_score": z_score,
                "p_value": p_value
            })
            print(f"{model_names[i]} vs {model_names[j]}: z={z_score:.4f}, p={p_value:.4f}")
    
    # 保存结果
    if save_path:
        pd.DataFrame(comparisons).to_csv(f"{save_path}/delong_test_results.csv", index=False)
    return comparisons