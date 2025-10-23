# main.py
import os
import sys
import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pickle
import matplotlib.pyplot as plt
import glob
import glob  # 用于文件匹配
import shutil  # 用于文件复制

sys.path.append("/path/to/your/")
from feature_selection import select_features, remove_highly_correlated_features
from model_training import train_model
from model_evaluation import compare_models_delong, evaluate_model
from model_tuning import tune_model
from model_explanation import explain_model
from visualization import plot_cv_roc_curve

import sys
import os
from datetime import datetime


class TeeOutput:
    """同时将输出发送到终端和文件"""

    def __init__(self, file_path, mode="a"):
        self.terminal = sys.stdout
        self.file = open(file_path, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_output_logging(save_path):
    """设置输出记录"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_path, f"terminal_output_{timestamp}.log")

    # 重定向标准输出和标准错误
    sys.stdout = TeeOutput(log_file)
    sys.stderr = TeeOutput(log_file, "a")

    print(f"======= 开始记录终端输出 [{timestamp}] =======")
    return log_file


def setup_logging(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.basicConfig(
        filename=f"{save_path}/execution.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def preprocess_data(data):
    # 检查是否存在 'Diagnosis' 列
    if "Diagnosis" in data.columns:
        # 将Diagnosis列编码为二分类变量（GA=0，GS=1）
        data["Diagnosis_Encoded"] = (data["Diagnosis"] == "GS").astype(int)
    else:
        print("警告：数据中不包含 'Diagnosis' 列，跳过相关处理。")

    # 确保Group列存在
    if "Group" not in data.columns:
        raise ValueError("数据中缺少 'Group' 列，请检查数据文件")
        # 这里假设Group列是二分类变量，GA=0，GS=1
        # 如果确实没有Group列，您需要根据实际情况创建
        # 这里无法自动创建，因为我们不知道哪些是发癌组

    # 检查数据中是否存在 'Age' 列，若存在则转换为二分类变量
    if "Age" in data.columns:
        data["Age"] = (data["Age"] >= 50).astype(int)
    else:
        print("警告：数据中不包含 'Age' 列，跳过年龄转换。")

    # 标准化血清学指标
    serum_markers = ["PGI", "PGII", "PGI/II", "HP", "G17"]
    existing_markers = [marker for marker in serum_markers if marker in data.columns]

    if existing_markers:
        scaler = StandardScaler()
        data[existing_markers] = scaler.fit_transform(data[existing_markers])

    # 归一化深度学习参数
    scaler_minmax = MinMaxScaler()
    deep_learning_features = [
        "Total Patches",
        "MUC5AC Count",
        "MUC5AC Ratio",
        "MUC6 Count",
        "MUC6 Ratio",
        "MUC2 Count",
        "MUC2 Ratio",
        "CD10 Count",
        "CD10 Ratio",
        "Complete Type I Count",
        "Complete Type I Ratio",
        "Incomplete Type II Count",
        "Incomplete Type II Ratio",
        "Incomplete Type III Count",
        "Incomplete Type III Ratio",
    ]

    # 检查并选择存在的特征
    existing_features = [feature for feature in deep_learning_features if feature in data.columns]
    if existing_features:
        data[existing_features] = scaler_minmax.fit_transform(data[existing_features])

    # 打印数据预览
    logging.info("标准化和归一化后的数据预览：")
    logging.info(data.head())

    return data


def check_cv_aucs(model, X, y, cv=5):
    """
    检查交叉验证中是否有AUC=1的情况
    返回: (bool, list) - 第一个元素指示是否有AUC=1的情况，第二个元素是所有折的AUC值列表
    """
    cv_obj = StratifiedKFold(n_splits=cv, random_state=42, shuffle=True)
    aucs = []

    X = np.array(X)
    y = np.array(y)

    for train, test in cv_obj.split(X, y):
        model.fit(X[train], y[train])
        y_prob = model.predict_proba(X[test])[:, 1]
        fpr, tpr, _ = roc_curve(y[test], y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 检查AUC是否为1或非常接近1
        if roc_auc > 0.999:
            return True, aucs

    return False, aucs


def adjust_model_complexity(model, C_value):
    """根据当前C值调整模型复杂度，返回新的C值"""
    # 每次减少C值，增加正则化强度
    new_C = C_value * 0.8
    if new_C < 0.01:  # 设置一个最小阈值
        new_C = 0.01
    return new_C


def evaluate_feature_subsets(data, save_path):
    """
    使用select_features得到前10个候选特征，然后分别使用前1,2,...,10个特征构建模型
    """
    # 确保传递正确的数据集给特征选择函数
    # 先删除非数值列如'Diagnosis'
    analysis_data = data.copy()

    # 调用特征选择函数，同时指定最多纳入10个特征
    selected_features = select_features(analysis_data, save_path=save_path, max_features=10)
    print("选出的前10个特征列表：", selected_features)

    # 移除高度相关的特征
    selected_features = remove_highly_correlated_features(data, selected_features, threshold=0.7)
    print("移除高度相关特征后的特征列表：", selected_features)

    best_auc = -1
    best_subset = None

    # 依次尝试采用前1～10个特征建模
    for k in range(1, len(selected_features) + 1):
        features_subset = selected_features[:k]
        print(f"\n使用前 {k} 个特征: {features_subset}")
        X_subset = data[features_subset]
        y = data["Group"]

        # 训练模型（每次内部会做数据划分，random_state已固定确保可重复）
        model, X_test_scaled, y_test = train_model(X_subset, y)
        # 评估模型（返回准确率、AUC和混淆矩阵）
        accuracy, auc_score, conf_matrix = evaluate_model(model, X_test_scaled, y_test, save_path=save_path)
        print(f"子集特征数量 {k}，模型AUC: {auc_score:.4f}")

        if auc_score > best_auc:
            best_auc = auc_score
            best_subset = features_subset.copy()

    print(f"\n最佳模型使用 {len(best_subset)} 个特征，AUC 为 {best_auc:.4f}，特征列表: {best_subset}")
    return best_subset, best_auc


def build_optimal_model(data, save_path, model_name):
    """
    为单个数据集构建最优模型，并确保交叉验证中没有AUC=1的情况
    """
    print(f"\n\n=====================================================")
    print(f"开始构建模型: {model_name}")
    print(f"=====================================================\n")

    # 设置日志记录
    setup_logging(save_path)

    # 数据预处理
    data = preprocess_data(data)

    # 记录样本分布
    if "Diagnosis" in data.columns:
        logging.info(f"总样本数: {len(data)}")
        logging.info(f"GA样本数: {sum(data['Diagnosis'] == 'GA')}")
        logging.info(f"GS样本数: {sum(data['Diagnosis'] == 'GS')}")
    else:
        logging.info("数据中不包含 'Diagnosis' 列，跳过样本分布统计。")

    # 特征选择
    best_features, best_auc = evaluate_feature_subsets(data, save_path=save_path)
    if len(best_features) == 0:
        raise ValueError("特征选择结果为空，无法构建模型")

    # 使用最佳特征子集训练最终模型
    print("\n使用最佳特征子集构建的最终模型：")
    X_final = data[best_features]
    y_final = data["Group"]

    # 初始参数设置
    C_value = 1.0
    max_attempts = 10
    attempt = 0

    # 循环尝试不同的正则化强度，直到没有AUC=1的情况
    while attempt < max_attempts:
        # 创建模型
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver="saga", penalty="l1", C=C_value, max_iter=10000, random_state=42)

        # 检查交叉验证中是否有AUC=1的情况
        has_perfect_auc, aucs = check_cv_aucs(model, X_final, y_final)

        if has_perfect_auc:
            print(f"发现AUC=1的情况，当前C值: {C_value}，调整模型复杂度...")
            C_value = adjust_model_complexity(model, C_value)
            attempt += 1
        else:
            print(f"交叉验证AUC值: {aucs}")
            print(f"最终模型C值: {C_value}")
            break

    if attempt >= max_attempts:
        print(f"警告: 经过{max_attempts}次尝试后，仍然存在AUC=1的情况")

    # 使用最终确定的参数训练模型
    model = LogisticRegression(solver="saga", penalty="l1", C=C_value, max_iter=10000, random_state=42)
    model, X_test_scaled, y_test = train_model(X_final, y_final, C=C_value)

    # 评估模型
    accuracy, auc_score, conf_matrix = evaluate_model(model, X_test_scaled, y_test, save_path=save_path)
    logging.info("最终模型评估:")
    logging.info(f"准确率: {accuracy:.4f}")
    logging.info(f"AUC: {auc_score:.4f}")

    # 模型调优、解释和可视化
    best_model, best_params = tune_model(X_final, y_final, model)
    logging.info(f"最佳参数: {best_params}")

    best_model_tuned, _, _ = train_model(X_final, y_final, C=best_params["C"])
    best_accuracy, best_auc, best_conf_matrix = evaluate_model(
        best_model_tuned, X_test_scaled, y_test, save_path=save_path
    )

    logging.info(f"调优后模型准确率: {best_accuracy:.4f}")
    logging.info(f"调优后模型AUC: {best_auc:.4f}")

    explain_model(
        best_model_tuned,
        X_final.values,  # 确保传入numpy数组
        feature_names=X_final.columns.tolist(),
        save_path=save_path,
    )
    plot_cv_roc_curve(model, X_final, y_final, save_path=save_path)
    plot_cv_roc_curve(best_model_tuned, X_final, y_final, save_path=save_path)

    # 返回模型和相关信息，添加了特征数据和目标变量
    return {
        "model": best_model_tuned,
        "model_name": model_name,  # 新增此行
        "features": best_features,
        "features_data": X_final,  # 添加特征数据
        "target_data": y_final,  # 添加目标变量
        "auc": best_auc,
        "params": best_params,
        "accuracy": best_accuracy,
    }


def evaluate_models_with_cv(models_results, model_names, save_path=None):
    """
    对多个模型进行五折交叉验证评估，比较平均AUC并检测AUC=1的情况

    参数:
    - models_results: 模型结果列表
    - model_names: 模型名称列表
    - save_path: 保存结果的路径
    """
    print("\n\n=== 五折交叉验证评估结果 ===")

    all_cv_results = []

    for i, (result, model_name) in enumerate(zip(models_results, model_names)):
        model = result["model"]
        X = result["features_data"]  # 需要存储模型使用的特征数据
        y = result["target_data"]  # 需要存储目标变量

        # 执行五折交叉验证
        has_perfect_auc, aucs = check_cv_aucs(model, X, y, cv=5)

        # 计算平均AUC和标准差
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        print(f"\n{model_name}:")
        print(f"  - 五折交叉验证平均AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  - 各折AUC值: {[f'{auc:.4f}' for auc in aucs]}")

        if has_perfect_auc:
            print(f"  - 警告: 存在AUC=1或接近1的情况!")

        # 存储结果用于比较
        all_cv_results.append(
            {
                "model_name": model_name,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "aucs": aucs,
                "has_perfect_auc": has_perfect_auc,
            }
        )

    # 按平均AUC降序排序
    all_cv_results.sort(key=lambda x: x["mean_auc"], reverse=True)

    # 比较结果
    print("\n=== 模型交叉验证AUC比较（降序排列）===")
    for i, result in enumerate(all_cv_results):
        warning = " (存在AUC=1!)" if result["has_perfect_auc"] else ""
        print(f"{i+1}. {result['model_name']}: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}{warning}")

    return all_cv_results


def save_best_model_results(models_results, cv_results, model_names, save_path, original_paths=None):
    """
    保存模型比较结果和所有模型的详细信息

    参数:
    - models_results: 模型结果列表
    - cv_results: 交叉验证结果列表
    - model_names: 模型名称列表
    - save_path: 保存路径
    - original_paths: 原始模型结果的路径列表，用于复制特征选择图表
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 检查联合模型是否优于其他模型
    combined_model_idx = model_names.index("联合模型") if "联合模型" in model_names else -1

    # 按平均AUC降序排序有效结果(不含AUC=1)
    valid_results = [r for r in cv_results if not r["has_perfect_auc"]]
    valid_results.sort(key=lambda x: x["mean_auc"], reverse=True)

    # 联合模型是否为最佳模型
    is_combined_model_best = False
    if valid_results and combined_model_idx >= 0:
        best_model_name = valid_results[0]["model_name"]
        is_combined_model_best = best_model_name == "联合模型"

    # 创建总结报告
    with open(f"{save_path}/model_comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("=== 模型比较总结 ===\n\n")

        # 交叉验证结果
        f.write("五折交叉验证结果（按平均AUC降序排列）:\n")
        for i, result in enumerate(valid_results):
            warning = " (存在AUC=1!)" if result["has_perfect_auc"] else ""
            f.write(f"{i+1}. {result['model_name']}: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}{warning}\n")
            f.write(f"   各折AUC: {[f'{auc:.4f}' for auc in result['aucs']]}\n")

        f.write("\n")

        # 联合模型效能分析
        if combined_model_idx >= 0:
            f.write("联合模型效能分析:\n")
            if is_combined_model_best:
                f.write("✓ 联合模型的效能优于其他两种模型\n")
            else:
                best_idx = model_names.index(valid_results[0]["model_name"])
                f.write(f"✗ 联合模型的效能不是最佳，最佳模型是: {model_names[best_idx]}\n")

            # 计算联合模型与其他模型的AUC差异
            combined_auc = next(r["mean_auc"] for r in valid_results if r["model_name"] == "联合模型")
            for result in valid_results:
                if result["model_name"] != "联合模型":
                    auc_diff = combined_auc - result["mean_auc"]
                    f.write(f"   联合模型比{result['model_name']}的AUC高: {auc_diff:.4f}\n")

        f.write("\n")

        # 各模型详细信息
        f.write("各模型详细信息:\n")
        for i, (result, model_name) in enumerate(zip(models_results, model_names)):
            f.write(f"\n{model_name}:\n")
            f.write(f"  特征数量: {len(result['features'])}\n")
            f.write(f"  特征列表: {result['features']}\n")
            f.write(f"  最终AUC: {result['auc']:.4f}\n")
            f.write(f"  准确率: {result['accuracy']:.4f}\n")
            f.write(f"  最佳参数: {result['params']}\n")

            if hasattr(result["model"], "coef_"):
                coefs = result["model"].coef_[0]
                feature_importance = pd.DataFrame({"特征": result["features"], "权重": coefs})
                feature_importance = feature_importance.sort_values(by="权重", key=abs, ascending=False)
                f.write(f"  特征权重(按绝对值大小排序):\n")
                for idx, row in feature_importance.iterrows():
                    f.write(f"    {row['特征']}: {row['权重']:.4f}\n")

    # 保存所有模型的ROC曲线
    for i, (result, model_name) in enumerate(zip(models_results, model_names)):
        model_save_path = os.path.join(save_path, f"{model_name}")
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        # 保存ROC曲线
        plot_cv_roc_curve(result["model"], result["features_data"], result["target_data"], save_path=model_save_path)

        # 保存模型的特征重要性可视化
        if hasattr(result["model"], "coef_"):
            plt.figure(figsize=(10, 8))
            coefs = result["model"].coef_[0]
            feature_importance = pd.DataFrame({"特征": result["features"], "权重": coefs})
            feature_importance = feature_importance.sort_values(by="权重", key=abs, ascending=False)

            plt.barh(feature_importance["特征"], abs(feature_importance["权重"]))
            plt.title(f"{model_name} - Feature importance")
            plt.tight_layout()
            plt.savefig(f"{model_save_path}/feature_importance.png")
            plt.close()

        # 保存模型对象
        with open(f"{model_save_path}/model.pkl", "wb") as f:
            pickle.dump(result["model"], f)

        # 保存特征列表
        with open(f"{model_save_path}/features.txt", "w", encoding="utf-8") as f:
            f.write(f"模型: {model_name}\n")
            f.write(f"特征列表: {', '.join(result['features'])}\n")
            f.write(f"AUC: {result['auc']:.4f}\n")

    # 保存最佳模型（无AUC=1情况）
    if valid_results:
        best_model_name = valid_results[0]["model_name"]
        best_model_idx = model_names.index(best_model_name)

        # 获取最佳模型
        best_model = models_results[best_model_idx]["model"]
        best_features = models_results[best_model_idx]["features"]

        # 保存到best_model文件夹
        best_model_dir = os.path.join(save_path, "best_model")
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)

        # 保存最佳模型对象
        with open(f"{best_model_dir}/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        # 保存特征列表
        with open(f"{best_model_dir}/best_features.txt", "w", encoding="utf-8") as f:
            f.write(f"最佳模型: {best_model_name}\n")
            f.write(f"特征列表: {', '.join(best_features)}\n")
            f.write(f"AUC: {models_results[best_model_idx]['auc']:.4f}\n")
            f.write(f"交叉验证平均AUC: {valid_results[0]['mean_auc']:.4f} ± {valid_results[0]['std_auc']:.4f}\n")

        print(f"\n所有模型结果已保存到: {save_path}")
        print(f"最佳模型 '{best_model_name}' 的详细信息已保存到: {best_model_dir}")

        # 复制特征选择阶段的图表
        if original_paths:
            for i, (path, model_name) in enumerate(zip(original_paths, model_names)):
                # 创建目标目录
                feature_selection_dir = os.path.join(save_path, f"{model_name}/feature_selection")
                if not os.path.exists(feature_selection_dir):
                    os.makedirs(feature_selection_dir)

                # 复制所有特征选择相关图片
                import shutil

                src_files = []
                for ext in ["*.png", "*.jpg", "*.pdf"]:
                    src_files.extend(glob.glob(os.path.join(path, ext)))

                for file in src_files:
                    # 只复制特征选择相关的图表，排除ROC曲线等
                    if any(
                        keyword in os.path.basename(file).lower()
                        for keyword in ["feature", "importance", "lasso", "forest", "correlation"]
                    ):
                        shutil.copy2(file, feature_selection_dir)
                        print(f"复制特征选择图表: {file} -> {feature_selection_dir}")

        return best_model_name, valid_results[0]["mean_auc"]
    else:
        print("警告: 所有模型都存在AUC=1的情况，无法确定最佳模型")
        return None, None


def main():
    # 设置终端输出记录
    log_file = setup_output_logging("/path/to/your/logs")
    print(f"终端输出将被记录到: {log_file}")

    # 定义三个数据集路径
    data_paths = [
        "/path/to/your/data/data_1.xlsx",
        "/path/to/your/data/data_2.xlsx",
        "/path/to/your/data/data_3.xlsx",
    ]

    # 定义对应的保存路径和模型名称
    save_paths = [
        "/path/to/your/results/results_1_new",
        "/path/to/your/results/results_2_new",
        "/path/to/your/results/results_3_new",
    ]

    model_names = ["Integrated model", "Pathological model", "Clinical model"]

    # 存储所有模型结果
    models_results = []

    # 循环构建三个模型
    models_results = []
    for i, (data_path, save_path, model_name) in enumerate(zip(data_paths, save_paths, model_names)):
        try:
            data = pd.read_excel(data_path)
            print(f"\n处理数据集 {i+1}/3: {model_name} - {data_path}")
            model_result = build_optimal_model(data, save_path, model_name)
            models_results.append(model_result)
            print(f"成功构建模型 {model_name}")
        except Exception as e:
            print(f"\n[严重错误] 模型 {model_name} 构建失败:")
            print(f"错误文件: {data_path}")
            print(f"错误详情: {str(e)}")
            logging.error(f"模型 {model_name} 构建失败: {str(e)}")
            continue  # 跳过当前模型，继续构建下一个

    # === 新增验证 ===
    print("\n最终构建的模型列表:")
    for model_result in models_results:
        print(f"- {model_result['model_name']} (存在)")

    # 打印所有模型的比较结果
    print("\n\n=== 模型比较 ===")
    for i, (result, model_name) in enumerate(zip(models_results, model_names)):
        print(f"{model_name}: AUC={result['auc']:.4f}, 准确率={result['accuracy']:.4f}, 特征={result['features']}")
    # Delong检验
    best_model_save_path = "/path/to/your/results"
    delong_results = compare_models_delong(models_results, model_names, save_path=best_model_save_path)

    # 添加交叉验证评估
    cv_results = evaluate_models_with_cv(models_results, model_names)

    # 保存最佳模型结果
    best_model_save_path = "/path/to/your/results"
    save_best_model_results(
        models_results, cv_results, model_names, best_model_save_path, original_paths=save_paths
    )  # 传递原始保存路径

    print("\n模型构建完成!")

    print("\n======= 终端记录完成 =======")
    if hasattr(sys.stdout, "close"):
        sys.stdout.close()
    if hasattr(sys.stderr, "close"):
        sys.stderr.close()


if __name__ == "__main__":
    main()
