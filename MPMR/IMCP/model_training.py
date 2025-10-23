# model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def train_model(X, y, test_size=0.3, random_state=42, C=1.0):
    """
    训练逻辑回归模型，支持自定义正则化强度C

    参数:
        X: 特征矩阵
        y: 目标变量
        test_size: 测试集比例，默认0.3
        random_state: 随机种子，默认42
        C: 正则化强度的倒数，默认1.0，值越小正则化越强
    """
    # 划分训练集和测试集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 训练Logistic回归模型
    model = LogisticRegression(solver="saga", penalty="l1", C=C, max_iter=10000, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # 早停法
    best_auc = 0
    best_model = None
    for i in range(1, 10001, 100):
        model.max_iter = i
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_val_pred)
        if auc > best_auc:
            best_auc = auc
            best_model = model
        else:
            break

    return best_model, X_val_scaled, y_val
