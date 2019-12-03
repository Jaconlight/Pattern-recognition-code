import numpy as np
import pandas as pd
def main(df_train, df_predict):
    beta = 1e-9
    x_train = df_train.iloc[:, :18]
    y_train = df_train.iloc[:, -1]
    b = np.ones([x_train.shape[1] + 1, 1])
    x = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    J_pre = np.dot((y_train - np.squeeze(np.dot(x, b))).T, (y_train - np.squeeze(np.dot(x, b)))) / len(y_train)
    diff = np.dot(x.T, y_train) - np.squeeze(np.dot(np.dot(x.T, x), b))
    b = np.squeeze(b) + beta * diff
    J_af = np.dot((y_train - np.squeeze(np.dot(x, b))).T, (y_train - np.squeeze(np.dot(x, b)))) / len(y_train)
    while np.sqrt(np.sum(np.square(J_af - J_pre))) >= 1e+5:
        J_pre = J_af
        diff = np.dot(x.T, y_train) - np.squeeze(np.dot(np.dot(x.T, x), b))
        b = np.squeeze(b) + beta * diff
        J_af = np.dot((y_train - np.squeeze(np.dot(x, b))).T, (y_train - np.squeeze(np.dot(x, b)))) / len(y_train)
    x_predict = df_predict.iloc[:, 1:]
    x_ = np.hstack((np.ones((x_predict.shape[0], 1)), x_predict))
    y_predict = np.dot(x_, b)
    rmse = np.sqrt(np.sum(np.square(np.dot(x, b) - y_train)) / len(y_train))
    return y_predict, rmse, b

if __name__ == '__main__':
    data_train = pd.read_csv(r'train.csv')
    for col in list(data_train.columns[data_train.isnull().sum() > 0]):
        mean_val = data_train[col].mean()
        data_train[col] = data_train[col].fillna(mean_val)
    data_predict = pd.read_csv(r'test.csv')
    y_predict, rmse, b = main(data_train, data_predict)
    dataframe = pd.DataFrame(y_predict, columns=["prediction"])
    dataframe.to_csv(r"gradient_descent_prediction.csv")
    print("RMSE：{}, 系数：{}".format(rmse, b))