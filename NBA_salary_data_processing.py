import pandas as pd
import matplotlib.pyplot as plt
# 以下代码分别运行
# 缺失数据
data_train = pd.read_csv(r'train.csv')
empty_index = data_train.index[data_train.isnull().sum(axis=1) > 0]
data = data_train.iloc[empty_index, :]
print(data)

# 对缺失数据进行填充
data_train = pd.read_csv(r'train.csv')
empty_index = data_train.index[data_train.isnull().sum(axis=1) > 0]
for col in list(data_train.columns[data_train.isnull().sum() > 0]):
    mean_val = data_train[col].mean()
    data_train[col] = data_train[col].fillna(mean_val)
data = data_train.iloc[empty_index, :]
print(data)

# 相关系数
data_train = pd.read_csv(r'train.csv')
for col in list(data_train.columns[data_train.isnull().sum() > 0]):
    mean_val = data_train[col].mean()
    data_train[col] = data_train[col].fillna(mean_val)
data_corr = data_train.corr()
print(data_corr.iloc[:-1, -1])

# FG、FGA、PTS与薪水的散布图
data_train = pd.read_csv(r'train.csv')
for col in list(data_train.columns[data_train.isnull().sum() > 0]):
    mean_val = data_train[col].mean()
    data_train[col] = data_train[col].fillna(mean_val)

data_salary = data_train['Salary']
data_FG = data_train['FG']
data_FGA = data_train['FGA']
data_PTS = data_train['PTS']
plt.figure(1)
plt.scatter(data_FG, data_salary, c='r')
plt.xlabel('FG')
plt.ylabel('Salary')
plt.figure(2)
plt.scatter(data_FGA, data_salary, c='b')
plt.xlabel('FGA')
plt.ylabel('Salary')
plt.figure(3)
plt.scatter(data_PTS, data_salary, c='g')
plt.xlabel('PTS')
plt.ylabel('Salary')
plt.show()