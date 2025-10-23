# visualization.py

import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np


def plot_cv_roc_curve(model, X, y, cv=5, save_path=None):
    # 使用StratifiedKFold进行交叉验证
    cv = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    X = np.array(X)  # 确保X是Numpy数组
    y = np.array(y)  # 确保y是Numpy数组

    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train])
        y_prob = model.predict_proba(X[test])[:, 1]
        fpr, tpr, _ = roc_curve(y[test], y_prob)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"ROC fold {i} (AUC = {roc_auc:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.")

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Cross-Validated Receiver Operating Characteristic", fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/cv_roc_curve.png")
    plt.close()
