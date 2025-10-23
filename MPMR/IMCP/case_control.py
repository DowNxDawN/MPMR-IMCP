# 导入必要库
import pandas as pd
import numpy as np
import os

# 数据加载与预处理
input_folder = "/path/to/your/data"
case_df = pd.read_excel(os.path.join(input_folder, "case_data.xlsx"))  # 病例组
ctrl_df = pd.read_excel(os.path.join(input_folder, "control_data.xlsx"))  # 对照组

# 处理缺失值（删除含缺失样本）
case_df = case_df.dropna()
ctrl_df = ctrl_df.dropna()

# 编码分类变量（性别）
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in ["Sex"]:
    case_df[col] = le.fit_transform(case_df[col])
    ctrl_df[col] = le.transform(ctrl_df[col])


# 1:1最近邻匹配（基于年龄和性别）
def case_control_matching(case_df, ctrl_df, match_vars, ratio=1):
    # 添加组标识
    case_df = case_df.copy()
    ctrl_df = ctrl_df.copy()
    case_df["group"] = 1
    ctrl_df["group"] = 0

    matched_pairs = []
    available_controls = ctrl_df.index.tolist()

    for i, case in case_df.iterrows():
        if len(available_controls) < ratio:
            print(f"警告：对照不足，只能匹配{len(available_controls)}个")
            break

        # 计算与每个可用对照的距离
        case_values = np.array(case[match_vars]).reshape(1, -1)
        ctrl_values = np.array(ctrl_df.loc[available_controls, match_vars])

        # 对于性别这样的分类变量，给予更大权重
        weights = np.ones(len(match_vars))
        for j, var in enumerate(match_vars):
            if var == "Sex":  # 性别精确匹配
                weights[j] = 100  # 给性别很大的权重

        # 计算加权距离
        distances = np.sqrt(np.sum(weights * (ctrl_values - case_values) ** 2, axis=1))

        # 选择最近的ratio个对照
        nearest_indices = np.argsort(distances)[:ratio]
        matched_control_indices = [available_controls[idx] for idx in nearest_indices]

        # 从可用对照中移除已选择的对照
        for idx in matched_control_indices:
            available_controls.remove(idx)

        # 为每个匹配创建一个匹配组ID
        for ctrl_idx in matched_control_indices:
            matched_pairs.append({"case_idx": i, "control_idx": ctrl_idx, "match_id": len(matched_pairs) // ratio + 1})

    # 创建匹配结果数据框
    matched_data = pd.DataFrame()

    for pair in matched_pairs:
        # 获取病例和对照的数据
        case_row = case_df.loc[pair["case_idx"]].copy()
        ctrl_row = ctrl_df.loc[pair["control_idx"]].copy()

        # 添加匹配ID
        case_row["match_id"] = pair["match_id"]
        ctrl_row["match_id"] = pair["match_id"]

        # 合并到结果中
        matched_data = pd.concat([matched_data, pd.DataFrame([case_row, ctrl_row])])

    return matched_data


# 执行1:1匹配（只基于年龄和性别）
match_vars = ["Age", "Sex"]
matched_data = case_control_matching(case_df, ctrl_df, match_vars, ratio=2)  # 此处可修改配对比例

# 提取匹配后的对照组
matched_controls = matched_data[matched_data["group"] == 0]
print(f"匹配后的对照组样本量: {len(matched_controls)}")

# 导出匹配后的数据（保存在与输入文件相同的文件夹中）
matched_data.to_excel(os.path.join(input_folder, "匹配后完整数据.xlsx"), index=False)
matched_controls.to_excel(os.path.join(input_folder, "匹配后对照组.xlsx"), index=False)

print(f"匹配完成！结果已保存到: {input_folder}")
print(f"- 匹配后完整数据.xlsx：包含所有病例和配对的对照")
print(f"- 匹配后对照组.xlsx：只包含配对的对照")

# 打印匹配后的样本量统计
print("\n匹配结果统计:")
print(f"- 原始病例组: {len(case_df)}例")
print(f"- 原始对照组: {len(ctrl_df)}例")
print(f"- 匹配后病例: {sum(matched_data['group'])}例")
print(f"- 匹配后对照: {len(matched_controls)}例")
print(f"- 匹配率: {sum(matched_data['group'])/len(case_df)*100:.1f}%")
