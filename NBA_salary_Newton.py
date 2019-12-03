import numpy as np
import pandas as pd
def main(df_train, df_predict):
    x_train = df_train.iloc[:, :18]
    y_train = df_train.iloc[:, -1]
    x = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    b = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y_train))
    x_predict = df_predict.iloc[:, 1:]
    x_ = np.hstack((np.ones((x_predict.shape[0], 1)), x_predict))
    y_predict = np.dot(x_, b)
    rmse = np.sqrt(np.sum(np.square(np.dot(x, b) - y_train)) / len(y_train))
    return rmse, y_predict, b

if __name__ == '__main__':
    data_train = pd.read_csv(r'train.csv')
    for col in list(data_train.columns[data_train.isnull().sum() > 0]):
        mean_val = data_train[col].mean()
        data_train[col] = data_train[col].fillna(mean_val)
    data_predict = pd.read_csv(r'test.csv')
    rmse, y_predict, b = main(data_train, data_predict)
    dataframe = pd.DataFrame(y_predict, columns=["prediction"])
    dataframe.to_csv(r"newton_prediction.csv")
    print("RMSE：{}, 系数：{}".format(rmse, b))