# model_tuning.py

from sklearn.model_selection import GridSearchCV


def tune_model(X, y, model):
    # 设置超参数范围
    param_grid = {"C": [0.1, 1, 10], "penalty": ["l2"]}

    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", error_score="raise")
    grid_search.fit(X, y)

    # 打印每个参数组合的得分
    print("Grid Search Results:")
    for mean, std, params in zip(
        grid_search.cv_results_["mean_test_score"],
        grid_search.cv_results_["std_test_score"],
        grid_search.cv_results_["params"],
    ):
        print(f"Mean AUC: {mean:.4f} (Std: {std:.4f}) with: {params}")

    # 输出最佳参数
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {best_params}")

    return best_model, best_params
