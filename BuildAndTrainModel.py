import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

path = "train.csv"
# 读取处理后的数据
df = pd.read_csv(f'after_{path}')

# 删除编号列 'Id'
df.drop('Id', axis=1, inplace=True)
df.drop('SaleCondition', axis=1, inplace=True)

# 将数据分为特征变量 X 和目标变量 y
X = df.iloc[:, :-1]  # 除去最后一列 'SalePrice'
y = df.iloc[:, -1]   # 最后一列是目标变量 'SalePrice'

# 划分训练集，测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

# 进行预测
y_pred = rf_model.predict(X_test)

# 评估模型性能
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# 打印评估结果
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# 保存模型到文件
model_path = 'model.joblib'
joblib.dump(rf_model, model_path)

print(f'Model saved to {model_path}')