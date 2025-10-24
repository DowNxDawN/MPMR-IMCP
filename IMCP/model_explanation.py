import shap
import matplotlib.pyplot as plt
import numpy as np

def explain_model(model, X_train, feature_names, save_path=None):
    """增强的SHAP分析"""
    plt.figure(figsize=(12, 8))
    
    # 创建解释器
    explainer = shap.Explainer(model, X_train, feature_names=feature_names)
    shap_values = explainer(X_train)
    
    # 1. 汇总图（条形图）
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/shap_summary_bar.png", dpi=300)
    plt.close()
    
    # 2. 汇总图（散点图）
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/shap_summary_dot.png", dpi=300)
    plt.close()
    
    # 3. 单个样本解释（随机选择5个样本）
    plt.figure(figsize=(12, 8))
    sample_indices = np.random.choice(len(X_train), 5, replace=False)
    for idx in sample_indices:
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/shap_waterfall_sample{idx}.png", dpi=300)
        plt.close()
    
    # 4. 特征依赖图（对所有重要特征）
    top_features = np.abs(shap_values.values).mean(0).argsort()[-5:][::-1]  # 取前5重要特征
    for feat_idx in top_features:
        shap.dependence_plot(feat_idx, shap_values.values, X_train, 
                           feature_names=feature_names, show=False)
        plt.title(f"Feature Dependence: {feature_names[feat_idx]}")
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/shap_dependence_{feature_names[feat_idx]}.png", dpi=300)
        plt.close()
    
    # 保存SHAP值数据
    if save_path:
        np.savez(f"{save_path}/shap_values.npz", 
                values=shap_values.values,
                base_values=shap_values.base_values,
                data=X_train,
                feature_names=feature_names)