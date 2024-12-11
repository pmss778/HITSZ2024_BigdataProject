import pandas as pd
from sklearn.preprocessing import LabelEncoder

path = "test.csv"
df = pd.read_csv(path)

# 遍历每一列，根据数据类型进行填充
for column in df.columns:
    if df[column].isnull().any():  # 如果该列有缺失值
        if df[column].dtype == 'object':
            # 分类列用众数填充，若众数为空则用 'None' 填充
            mode_value = df[column].mode()
            if not mode_value.empty:
                df[column].fillna(mode_value[0], inplace=True)
            else:
                df[column].fillna('None', inplace=True)
        else:
            # 数值型列用中位数填充
            df[column].fillna(df[column].median(), inplace=True)
# 找出需要进行标签编码的分类变量（object 类型的列）
categorical_cols = df.select_dtypes(include=['object']).columns

# 初始化 LabelEncoder
label_encoder = LabelEncoder()

# 对每个分类变量的列进行标签编码
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

df.to_csv(f'after_{path}', index=False)



