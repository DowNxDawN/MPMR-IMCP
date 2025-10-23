# feature_selection.py
import scipy.stats as stats
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib
import matplotlib.colors as mcolors
import shap
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端，防止图形窗口弹出

# 设置Seaborn风格，并配置中文字体，后续英文的图形我们直接传入英文文本
sns.set(style="whitegrid")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 可显示中文（若后续传英文不受影响）
matplotlib.rcParams["axes.unicode_minus"] = False


####################################################
# 新增辅助函数：绘制渐变条形图，如果顶部值异常则采用断裂效果
def plot_gradient_bar(
    data,
    title="",
    xlabel="",
    ylabel="",
    save_path=None,
    filename="plot.png",
    ratio_threshold=2.0,
    break_axis="auto",
    extra_line=None,
):
    """
    绘制渐变色柱状图。如果最上面的值远大于第二大的值（默认阈值为2倍），则采用broken axis效果，
    避免最高的柱子占用过多空间。

    参数：
      data: pandas Series（索引为特征名称，值为对应数值），应已排序（降序或升序均可）
      title: 图表标题
      xlabel: x轴标签
      ylabel: y轴标签
      save_path: 文件保存路径（若提供）
      filename: 保存的文件名
      ratio_threshold: 判断异常值的阈值（默认当最高值 > 第二高值 * ratio_threshold时采用断裂效果）
      break_axis: 'upper', 'lower', 'auto' or None. For "auto", auto-detect based on data.
      extra_line: dict (optional). If provided, should contain keys:
           'value' (numeric), 'color', 'linestyle', 'linewidth', and 'label'.
           An extra horizontal line will be drawn at the given value.
    """
    # Use the bar order to generate gradient colors by linear interpolation from the Set2 base palette.
    n = len(data)
    base_colors = sns.color_palette("Set2")  # 基色列表
    # 构造线性分段的ColorMap
    set2_cmap = mcolors.LinearSegmentedColormap.from_list("set2_cmap", base_colors)
    # 根据条形的顺序（从左到右）计算颜色
    colors = [set2_cmap(i / (n - 1) if n > 1 else 0) for i in range(n)]

    indices = np.arange(n)

    # Determine break mode
    mode = None
    if break_axis == "auto":
        if n > 1:
            # 先检测上端极值（数据应按降序排列）
            if data.iloc[0] > ratio_threshold * data.iloc[1]:
                mode = "upper"
            # 检测下端极值（数据应按升序排列且数值为负时）
            elif data.iloc[0] < 0 and abs(data.iloc[0]) > ratio_threshold * abs(data.iloc[1]):
                mode = "lower"
            else:
                mode = None
        else:
            mode = None
    elif break_axis in ["upper", "lower"]:
        mode = break_axis
    else:
        mode = None

    # --------------------------
    # 上端断裂处理：用于数据中最高值过高的情况（数据按降序排序）
    if mode == "upper":
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 3]}
        )
        ax_top.bar(indices, data.values, color=colors, align="center")
        ax_bottom.bar(indices, data.values, color=colors, align="center")

        # 定义断裂临界值：采用第二高值乘以1.1
        threshold = data.iloc[1] * 1.1
        ax_top.set_ylim(threshold, data.iloc[0] * 1.05)
        ax_bottom.set_ylim(0, threshold)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        d = 0.005  # 斜线尺寸（坐标轴比例）
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_bottom.set_xticks(indices)
        ax_bottom.set_xticklabels(data.index, rotation=45, ha="right")
        ax_top.set_title(title, fontsize=16)
        ax_bottom.set_xlabel(xlabel, fontsize=12)
        ax_bottom.set_ylabel(ylabel, fontsize=12)

        # 添加额外的参考线（例如：P-value参考线）
        if extra_line:
            if extra_line["value"] > threshold:
                ax_top.axhline(
                    y=extra_line["value"],
                    color=extra_line.get("color", "lightgray"),
                    linestyle=extra_line.get("linestyle", "--"),
                    linewidth=extra_line.get("linewidth", 2),
                )
                ax_top.text(
                    len(data) * 0.95,
                    extra_line["value"] + 0.01 * (data.iloc[0]),
                    extra_line.get("label", ""),
                    color=extra_line.get("color", "lightgray"),
                )
            else:
                ax_bottom.axhline(
                    y=extra_line["value"],
                    color=extra_line.get("color", "lightgray"),
                    linestyle=extra_line.get("linestyle", "--"),
                    linewidth=extra_line.get("linewidth", 2),
                )
                ax_bottom.text(
                    len(data) * 0.95,
                    extra_line["value"] + 0.01 * (data.iloc[-1]),
                    extra_line.get("label", ""),
                    color=extra_line.get("color", "lightgray"),
                )
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/{filename}")
        plt.close()

    # --------------------------
    # 下端断裂处理：用于数据中最低值过低的情况（数据按升序排列）
    elif mode == "lower":
        # 确保数据按升序排列
        data = data.sort_values(ascending=True)
        indices = np.arange(len(data))
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, sharex=True, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
        ax_top.bar(indices, data.values, color=colors, align="center")
        ax_bottom.bar(indices, data.values, color=colors, align="center")

        # 定义下端断裂的临界值：取第二低值乘以0.9（数值为负时，该值会变得不那么低）
        threshold = data.iloc[1] * 0.9
        ax_bottom.set_ylim(data.iloc[0] * 1.05, threshold)
        ax_top.set_ylim(threshold, data.max() * 1.05)

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)

        d = 0.005
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
        ax_top.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (-d, -d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (-d, -d), **kwargs)

        ax_bottom.set_xticks(indices)
        ax_bottom.set_xticklabels(data.index, rotation=45, ha="right")
        ax_top.set_title(title, fontsize=16)
        ax_bottom.set_xlabel(xlabel, fontsize=12)
        ax_bottom.set_ylabel(ylabel, fontsize=12)

        if extra_line:
            if extra_line["value"] < threshold:
                ax_bottom.axhline(
                    y=extra_line["value"],
                    color=extra_line.get("color", "lightgray"),
                    linestyle=extra_line.get("linestyle", "--"),
                    linewidth=extra_line.get("linewidth", 2),
                )
                ax_bottom.text(
                    len(data) * 0.95,
                    extra_line["value"] + 0.01 * (data.iloc[0]),
                    extra_line.get("label", ""),
                    color=extra_line.get("color", "lightgray"),
                )
            else:
                ax_top.axhline(
                    y=extra_line["value"],
                    color=extra_line.get("color", "lightgray"),
                    linestyle=extra_line.get("linestyle", "--"),
                    linewidth=extra_line.get("linewidth", 2),
                )
                ax_top.text(
                    len(data) * 0.95,
                    extra_line["value"] + 0.01 * (data.iloc[-1]),
                    extra_line.get("label", ""),
                    color=extra_line.get("color", "lightgray"),
                )
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/{filename}")
        plt.close()

    # --------------------------
    # 普通绘制（无断裂效果）
    else:
        plt.figure(figsize=(12, 6))
        plt.bar(indices, data.values, color=colors, align="center")
        plt.xticks(indices, data.index, rotation=45, ha="right")
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        if extra_line:
            plt.axhline(
                y=extra_line["value"],
                color=extra_line.get("color", "lightgray"),
                linestyle=extra_line.get("linestyle", "--"),
                linewidth=extra_line.get("linewidth", 2),
            )
            plt.text(
                len(data) * 0.95,
                extra_line["value"] + 0.01 * data.values.max(),
                extra_line.get("label", ""),
                color=extra_line.get("color", "lightgray"),
            )
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/{filename}")
        plt.close()


####################################################
# 新增函数：用于绘制 SHAP Value 图（横向条形图）并在横轴 -1.5 到 -3.5 处截断
def plot_shap_gradient_bar(
    shap_series, title="", xlabel="", ylabel="", custom_break=(-3.5, -1.5), save_path=None, filename="shap_values.png"
):
    """
    针对 SHAP Value 图绘制横向渐变条形图，并在横轴 custom_break 指定的范围内做断裂效果，
    以避免极端值拉长整个图形。

    参数：
      shap_series: pandas Series，索引为特征名称，值为对应 SHAP 值，
                   请确保数据按升序排列（使极端值在最左侧）。
      title: 图表标题
      xlabel: x轴标签
      ylabel: y轴标签
      custom_break: 二元组，形如 (-3.5, -1.5)，表示需要截断的 x 轴区域
      save_path: 如指定，则保存图像到该目录
      filename: 保存文件名
    """
    # 对数据按升序排序
    shap_series = shap_series.sort_values(ascending=True)
    features = shap_series.index.tolist()
    values = shap_series.values
    n = len(shap_series)

    # 生成渐变色（使用 Set2 调色板）
    base_colors = sns.color_palette("Set2")
    set2_cmap = mcolors.LinearSegmentedColormap.from_list("set2_cmap", base_colors)
    colors = [set2_cmap(i / (n - 1) if n > 1 else 0) for i in range(n)]

    # 创建左右两个子图，左右共享 y 轴；左侧显示极端值部分，右侧显示其余部分
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3], wspace=0.05)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    # 定义左右子图的 x 轴范围
    # 左侧显示：从最小值到 custom_break[0]；右侧显示：从 custom_break[1] 到最大值
    left_xlim = (min(values) * 1.05, custom_break[0])
    right_xlim = (custom_break[1], max(values) * 1.05)

    # 遍历每个条目，根据其 SHAP 值决定在哪个子图中绘制
    for i, (feat, val, col) in enumerate(zip(features, values, colors)):
        if val < custom_break[0]:
            ax_left.barh(i, val, color=col, align="center")
        else:
            ax_left.barh(i, 0, color=col, align="center")  # 用于保持对齐
        if val >= custom_break[1]:
            ax_right.barh(i, val, color=col, align="center")
        else:
            ax_right.barh(i, 0, color=col, align="center")

    # 设置 y 轴刻度及标签，并在左侧显示
    ax_left.set_yticks(range(n))
    ax_left.set_yticklabels(features)
    ax_left.invert_yaxis()  # 若需要倒序显示，可取消此行注释

    ax_left.set_xlim(left_xlim)
    ax_right.set_xlim(right_xlim)

    # 隐藏左右拼图间不必要的轴线
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # 绘制断裂标记
    d = 0.015
    kwargs = dict(transform=ax_left.transAxes, color="k", clip_on=False)
    ax_left.plot((1 - d, 1 + d), (0.5 - d, 0.5 + d), **kwargs)
    ax_left.plot((1 - d, 1 + d), (0.5 + d, 0.5 - d), **kwargs)
    kwargs.update(transform=ax_right.transAxes)
    ax_right.plot((-d, d), (0.5 - d, 0.5 + d), **kwargs)
    ax_right.plot((-d, d), (0.5 + d, 0.5 - d), **kwargs)

    ax_left.set_title(title, fontsize=16)
    ax_right.set_xlabel(xlabel, fontsize=12)
    ax_left.set_ylabel(ylabel, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(f"{save_path}/{filename}")
    plt.close()


####################################################
# 修改后的森林图绘制函数（英文版），内容同之前，无中文文本
def plot_forest(
    features,
    effects,
    errors,
    pvalues=None,
    title="",
    xlabel="Effect",
    ylabel="Features",
    save_path=None,
    filename_prefix="forest_plot",
    show_significance=True,
    ci_fold=1,
):
    """
    Improved forest plot function (in English).

    Parameters:
      features: list of feature names
      effects: effect sizes (e.g. regression coefficients or importance values)
      errors: standard errors (or error bars); if regression model, multiply by ci_fold for CI.
      pvalues: p-values for each feature (optional; used to display significance markers)
      title: plot title
      xlabel: x-axis label
      ylabel: y-axis label
      save_path: if provided, save the plot in specified path
      filename_prefix: file name prefix for saving
      show_significance: whether to display statistical significance markers (requires pvalues)
      ci_fold: multiplier for errors to draw confidence intervals (e.g. 1.96 for 95% CI)
    """
    plt.figure(figsize=(10, max(6, len(features) * 0.5)))
    y_positions = np.arange(len(features))

    colors = []
    stars = []
    for i, eff in enumerate(effects):
        if show_significance and pvalues is not None:
            p = pvalues[i]
            if eff < 0:
                col = "blue" if p < 0.05 else "lightblue"
            elif eff > 0:
                col = "orange" if p < 0.05 else "wheat"
            else:
                col = "gray"
            if p < 0.001:
                star = "***"
            elif p < 0.05:
                star = "*"
            else:
                star = ""
        else:
            col = "dodgerblue"
            star = ""
        colors.append(col)
        stars.append(star)

    for i, (feature, eff, err, col, star) in enumerate(zip(features, effects, errors, colors, stars)):
        plt.errorbar(
            eff,
            y_positions[i],
            xerr=err * ci_fold,
            fmt="o",
            color=col,
            ecolor="gray",
            capsize=4,
            elinewidth=2,
            markeredgewidth=1,
        )
        if star:
            offset = 0.03 if eff >= 0 else -0.03
            ha = "left" if eff >= 0 else "right"
            plt.text(eff + offset, y_positions[i], star, va="center", ha=ha, fontsize=12, color=col)

    plt.yticks(y_positions, features, fontsize=12)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="x", linestyle="--", color="lightgray", alpha=0.7)

    all_lower = [eff - err * ci_fold for eff, err in zip(effects, errors)]
    all_upper = [eff + err * ci_fold for eff, err in zip(effects, errors)]
    data_min = min(all_lower)
    data_max = max(all_upper)
    if abs(data_min) < 0.5 and abs(data_max) < 0.5:
        margin = 0.2
        plt.xlim(data_min - margin, data_max + margin)
    else:
        plt.xlim(min(-1, data_min - 0.1), max(1, data_max + 0.1))

    if show_significance and pvalues is not None:
        neg_sig = mlines.Line2D(
            [], [], color="blue", marker="o", linestyle="None", markersize=8, label="Negative (P<0.05)"
        )
        neg_nonsig = mlines.Line2D(
            [], [], color="lightblue", marker="o", linestyle="None", markersize=8, label="Negative (P>=0.05)"
        )
        pos_sig = mlines.Line2D(
            [], [], color="orange", marker="o", linestyle="None", markersize=8, label="Positive (P<0.05)"
        )
        pos_nonsig = mlines.Line2D(
            [], [], color="wheat", marker="o", linestyle="None", markersize=8, label="Positive (P>=0.05)"
        )
        plt.legend(handles=[neg_sig, neg_nonsig, pos_sig, pos_nonsig], loc="best", fontsize=10)
    else:
        feat_line = mlines.Line2D(
            [], [], color="dodgerblue", marker="o", linestyle="None", markersize=8, label="Feature Value"
        )
        plt.legend(handles=[feat_line], loc="best", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{filename_prefix}.png")
    plt.close()


####################################################
# 修改后的图形调用示例


def remove_highly_correlated_features(data, selected_features, threshold=0.7):
    corr_matrix = data[selected_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [feature for feature in selected_features if feature not in to_drop]


def select_features(data, save_path=None, max_features=10):
    """
    使用多种特征选择方法筛选重要特征，并返回排序后的特征列表

    参数:
        data: 包含特征和目标变量的DataFrame
        save_path: 可视化结果保存路径
        max_features: 返回的最大特征数量

    返回:
        list: 按重要性排序的特征列表
    """
    # 排除非数值类型的列和目标变量
    feature_cols = [
        col for col in data.columns if col not in ["ID", "Group"] and pd.api.types.is_numeric_dtype(data[col])
    ]

    X = data[feature_cols]
    y = data["Group"]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ANOVA单因素分析
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(f_classif, k="all")
    selector.fit(X_scaled, y)
    anova_scores = pd.Series(selector.scores_, index=X.columns)
    anova_pvalues = pd.Series(selector.pvalues_, index=X.columns)
    anova_scores_sorted = anova_scores.sort_values(ascending=False)

    # 使用渐变柱状图展示 ANOVA Scores，如果最高值异常则截断
    plot_gradient_bar(
        anova_scores_sorted,
        title="ANOVA Scores",
        xlabel="Features",
        ylabel="Score",
        save_path=save_path,
        filename="anova_scores.png",
    )
    plt.close()  # 确保关闭图形

    # LASSO回归
    from sklearn.linear_model import LassoCV, lasso_path

    lasso = LassoCV(cv=5, max_iter=10000)
    lasso.fit(X_scaled, y)
    lasso_selected = pd.Series(lasso.coef_, index=X.columns)
    lasso_coefficients_sorted = lasso_selected.sort_values(ascending=False)
    plot_gradient_bar(
        lasso_coefficients_sorted,
        title="LASSO Coefficients",
        xlabel="Features",
        ylabel="Coefficient",
        save_path=save_path,
        filename="lasso_coefficients.png",
    )
    plt.close()  # 确保关闭图形

    # 新增：LASSO 正则化路径图（美化版本）
    # 该图展示了各特征系数随正则化参数 alpha 变化的情况，并以不同颜色突出显示非零系数特征
    alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y, max_iter=10000)
    optimal_alpha = lasso.alpha_

    plt.figure(figsize=(14, 10))
    # 使用颜色映射：使用 tab20，为每个特征分配颜色
    cmap = plt.cm.get_cmap("tab20", len(X.columns))

    # 计算最佳系数：用于判断哪些特征在最优 alpha 下的系数非零
    best_coef = lasso.coef_
    nonzero_features = [i for i, coef in enumerate(best_coef) if abs(coef) > 1e-4]

    for i, feature in enumerate(X.columns):
        if i in nonzero_features:
            # 对于非零系数特征，绘制较宽的线条并添加标签
            plt.plot(alphas_lasso, coefs_lasso[i], label=feature, color=cmap(i), linewidth=2)
        else:
            # 其余特征绘制为灰色、较细且半透明的线条
            plt.plot(alphas_lasso, coefs_lasso[i], color="grey", linewidth=0.5, alpha=0.5)

    # 在图中添加一条垂直虚线，标识最优 alpha 值
    plt.axvline(
        x=optimal_alpha, linestyle="--", color="black", linewidth=1.5, label=f"Optimal alpha = {optimal_alpha:.4f}"
    )

    plt.xlabel("Alpha (log scale)", fontsize=14)
    plt.ylabel("Coefficient", fontsize=14)
    plt.title("LASSO Regularization Path", fontsize=16)
    plt.xscale("log")
    plt.tick_params(labelsize=12)

    # 仅显示有标签的非零项和最优 alpha的图例
    plt.legend(loc="best", fontsize=10, ncol=2, frameon=True, fancybox=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/lasso_regression_path.png", dpi=300)
    plt.close()

    # LASSO回归（OLS估计）系数森林图 —— 使用OLS获得真实参数（排除截距）
    X_with_const = sm.add_constant(X_scaled)
    ols_model = sm.OLS(y, X_with_const).fit()
    regression_effects = pd.Series(ols_model.params[1:], index=X.columns)
    regression_std_errors = pd.Series(ols_model.bse[1:], index=X.columns)
    regression_pvalues = pd.Series(ols_model.pvalues[1:], index=X.columns)
    plot_forest(
        features=regression_effects.index.tolist(),
        effects=regression_effects.values,
        errors=regression_std_errors.values,
        pvalues=regression_pvalues.values,
        title="LASSO Regression (OLS Estimate) Forest Plot",
        xlabel="Effect",
        ylabel="Features",
        save_path=save_path,
        filename_prefix="lasso_forest_plot",
        show_significance=True,
        ci_fold=1.96,
    )
    plt.close()  # 确保关闭图形

    # 随机森林特征重要性
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
    rf_importances_sorted = rf_importances.sort_values(ascending=False)
    plot_gradient_bar(
        rf_importances_sorted,
        title="Random Forest Importances",
        xlabel="Features",
        ylabel="Importance",
        save_path=save_path,
        filename="random_forest_importances.png",
    )
    plt.close()  # 确保关闭图形

    # 随机森林森林图 —— 不显示显著性
    all_importances = np.array([tree.feature_importances_ for tree in rf.estimators_])
    importances_std = np.std(all_importances, axis=0)
    plot_forest(
        features=rf_importances.index.tolist(),
        effects=rf_importances.values,
        errors=importances_std,
        title="Random Forest Feature Importance Forest Plot",
        xlabel="Importance",
        ylabel="Features",
        save_path=save_path,
        filename_prefix="random_forest_forest_plot",
        show_significance=False,
        ci_fold=1,
    )
    plt.close()  # 确保关闭图形

    # 逐步回归
    stepwise_model = sm.Logit(y, X_with_const).fit(maxiter=15000, tol=1e-8)
    stepwise_pvalues = pd.Series(stepwise_model.pvalues.values[1:], index=X.columns)
    stepwise_pvalues_sorted = stepwise_pvalues.sort_values(ascending=True)
    plot_gradient_bar(
        stepwise_pvalues_sorted,
        title="Stepwise Regression P-values",
        xlabel="Features",
        ylabel="P-value",
        save_path=save_path,
        filename="stepwise_pvalues.png",
        extra_line={"value": 0.05, "color": "lightgray", "linestyle": "--", "linewidth": 2},
    )
    plt.close()  # 确保关闭图形

    # 7. SHAP Value 图：采用 shap.summary_plot 绘制图，
    # 限制显示特征数量，避免极端值导致图表过长，同时调整图像尺寸与颜色映射
    # 为确保特征名称能正确显示，将 X_scaled 转换为 DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    import shap

    # 使用之前训练好的随机森林模型 rf 作为解释器，将 check_additivity 设置为 False
    explainer = shap.Explainer(rf, X_scaled_df)
    shap_values = explainer(X_scaled_df, check_additivity=False)
    # 绘制 SHAP Summary Plot：
    # max_display=10 限制最多显示 10 个特征；
    # 使用固定颜色映射 color="red"（您也可尝试其他颜色，如 "coolwarm"）；
    # 调整图像尺寸 plot_size=(12, 8) 来适应显示；
    # 传入 feature_names（转换为 numpy 数组）确保特征名称正确索引
    shap.summary_plot(
        shap_values,
        X_scaled_df.values,
        max_display=10,
        color="red",
        plot_size=(12, 8),
        feature_names=np.array(X_scaled_df.columns),
    )
    plt.close()  # 确保关闭图形

    # 合并各方法的特征评分（将 LASSO 部分替换为 OLS 结果）
    feature_scores = pd.DataFrame(
        {
            "ANOVA": anova_scores,
            "ANOVA P-value": anova_pvalues,
            "OLS": regression_effects,
            "OLS P-value": regression_pvalues,
            "RandomForest": rf_importances,
            "Stepwise": stepwise_pvalues,
        }
    )
    feature_scores = feature_scores.fillna(0)
    feature_scores["mean_score"] = feature_scores.mean(axis=1)

    # 使用max_features参数指定最终纳入的特征数
    selected_features = feature_scores.nlargest(max_features, "mean_score").index.tolist()
    print("Selected Features:")
    print(selected_features)

    # 去除强相关性特征
    selected_features = remove_highly_correlated_features(data, selected_features)
    print("Features after removing high correlation:")
    print(selected_features)

    # 进行相关性分析并绘制相关性矩阵图
    plot_correlation_matrix(data, selected_features, save_path=save_path)
    plt.close()  # 确保关闭图形

    return selected_features


def plot_correlation_matrix(data, selected_features, save_path=None):
    # 计算相关性矩阵
    corr_matrix = data[selected_features].corr()

    # 计算P值矩阵
    p_matrix = pd.DataFrame(np.zeros(corr_matrix.shape), columns=corr_matrix.columns, index=corr_matrix.index)
    for i in range(len(selected_features)):
        for j in range(len(selected_features)):
            if i != j:
                _, p_value = stats.pearsonr(data[selected_features[i]], data[selected_features[j]])
                p_matrix.iloc[i, j] = p_value

    # 格式化P值矩阵
    p_matrix_str = p_matrix.applymap(lambda x: f"P={x:.2f}" if x != 0 else "")
    p_matrix = p_matrix.applymap(lambda x: x if x != 0 else np.nan)

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只掩盖上三角部分，不包括对角线
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask, cbar_kws={"shrink": 0.8})
    sns.heatmap(p_matrix, annot=p_matrix_str, fmt="", cmap="coolwarm", vmin=0, vmax=1, mask=~mask, cbar=False)
    plt.title("Feature Correlation Matrix (Left: Correlation, Right: P-value)", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/correlation_matrix.png")
    plt.close()


from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt
import numpy as np


def plot_lasso_path(X, y, feature_names, save_path=None):
    """
    绘制 LASSO 回归路径图
    """
    # 计算 LASSO 路径
    alphas, coefs, _ = lasso_path(X, y, alphas=np.logspace(-4, 0, 100))

    # 绘图
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))  # 为每个特征分配不同颜色
    for coef, name, color in zip(coefs, feature_names, colors):
        plt.plot(alphas, coef, label=name, color=color)

    plt.xscale("log")
    plt.xlabel("Alpha (log scale)")
    plt.ylabel("Coefficients")
    plt.title("LASSO Path")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/lasso_path.png")
    plt.show()
