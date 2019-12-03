import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
def main(df_train, df_predict):
    x_train = df_train.iloc[:, :18]
    y_train = df_train.iloc[:, -1]
    linear_re = LinearRegression()
    linear_re.fit(x_train, y_train)
    x_predict = df_predict.iloc[:, 1:]
    y_predict = linear_re.predict(x_predict)
    rmse = np.sqrt(np.sum(np.square(linear_re.predict(x_train) - y_train)) / len(y_train))
    return y_predict, rmse, linear_re.coef_

if __name__ == '__main__':
    data_train = pd.read_csv(r'train.csv')
    for col in list(data_train.columns[data_train.isnull().sum() > 0]):
        mean_val = data_train[col].mean()
        data_train[col] = data_train[col].fillna(mean_val)
    data_predict = pd.read_csv(r'test.csv')
    y_predict, rmse, intercept = main(data_train, data_predict)
    dataframe = pd.DataFrame(y_predict, columns=["prediction"])
    dataframe.to_csv(r"sklearn_prediction.csv")
    print("RMSE：{}, 系数：{}".format(rmse, intercept))