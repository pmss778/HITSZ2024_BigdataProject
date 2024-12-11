import pandas as pd
import joblib

path = "test.csv"
# 读取处理后的数据
df = pd.read_csv(f'after_{path}')

# 删除编号列 'Id'
df.drop('Id', axis=1, inplace=True)

# 将数据分为特征变量 X（去掉目标变量 'SalePrice'）
X = df.iloc[:, :-1]  # 除去最后一列 'SalePrice'

# 加载保存的模型
model_path = 'model.joblib'
loaded_model = joblib.load(model_path)

# 使用加载的模型进行预测
predictions = loaded_model.predict(X)
DF = pd.read_csv(f'after_{path}')
results = pd.DataFrame({
    'Id': DF['Id'],  # 仍然保留 Id 列
    'SalePrice': predictions  # 新增预测结果列
})

# 保存结果到 CSV 文件
output_path = 'predictions.csv'
results.to_csv(output_path, index=False)

print(f'Predictions saved to {output_path}')
