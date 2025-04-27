import pandas as pd
from torch.utils.data import Dataset

from utils.timefeature import timefeature



class MyDataset(Dataset):
    def __init__(self, df, scaler, scaler1, seq_len=96, label_len=48, pred_len=24):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scaler = scaler
        self.scaler1 = scaler1

        self.__read_data__(df=df)

    def __read_data__(self, df):
        data = df.iloc[:, 1:5].values.reshape(len(df), -1)  # 归一化4列
        data = self.scaler.transform(data)
        data1 = df.iloc[:, 5].values.reshape(len(df), -1)  # 归一化最后一列
        data1 = self.scaler1.transform(data1)
        df["date"] = pd.to_datetime(df.iloc[:, 0])  # 日期列！
        stamp = timefeature(df)  # 需要吗？？

        self.data_x = data  # 特征是第2列
        self.data_y = data1  # 标签是第3列
        self.stamp = stamp
        # self.data_x = data  # 特征是第一列和第二列（去掉时间列后的第一列）
        # self.data_y = data  # 标签是第一列和第二列
        # self.stamp = stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]  # 特征序列
        seq_y = self.data_y[r_begin:r_end]  # 标签序列
        seq_x_mark = self.stamp[s_begin:s_end]  # 特征序列的时间标记，用于指示每个数据点的时间信息
        seq_y_mark = self.stamp[r_begin:r_end]  # 标签序列的时间标记

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1  # 根据原始行数计算得到训练集/测试集的样本数

