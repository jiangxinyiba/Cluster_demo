# 密度聚类方法 DBSCAN
import pandas as pd
import numpy as np
from pandas import *
from numpy import *
import matplotlib.pyplot as plt
import math

def createData():
    # 西瓜数据集4.0
    Data = [[0.697,0.46],
            [0.774,0.376],
            [0.634,0.264],
            [0.608,0.318],
            [0.556,0.215],
            [0.403,0.237],
            [0.481,0.149],
            [0.437,0.211],
            [0.666,0.091],
            [0.243,0.267],
            [0.245,0.057],
            [0.343,0.099],
            [0.639,0.161],
            [0.657,0.198],
            [0.36,0.37],
            [0.593,0.042],
            [0.719,0.103]]
    Data0 = np.array(Data)
    # using pandas dataframe for .csv read which contains chinese char.
    df0 = DataFrame(np.random.randn(5, 2), index=range(0, 10, 2), columns=list('AB'))

    with open("D:/Code/PycharmProjects/MachineLearning_demo/watermelon_3.csv", mode='r', encoding="gb18030" ) as data_file:
        df = pd.read_csv(data_file)
        # Data = df[1:]
        Data = df.loc[:,["密度","含糖率"]]
    return Data

def DetermineEArea(X,xj,epslion):
    distX = np.zeros(X.shape[0])
    EArea = []
    for i in range(X.shape[0]):
        xi = X.loc[i]
        distX[i] = dist(xi,xj)
    index = distX<epslion
    EArea = X.loc[index]
    N_EArea = EArea.shape[0]
    return EArea,N_EArea
def dist(xi,xj):
    detX = (xi-xj)*(xi-xj)
    distX = np.sqrt(detX[0]+detX[1])
    return distX

# 在X集合中去除Xdelte包含的行
def DeleteSample(X,Xdelete):
    # 判断这个Object在Data中的位置
    flag1 = X.密度.isin(Xdelete.密度)
    flag2 = X.含糖率.isin(Xdelete.含糖率)
    flag = flag1 & flag2
    # 在Data样本集中去除Q_Object中的核心对象
    X = X[-flag]
    return X

def DBSCAN(X,epslion,MinPts):
    # 初始化核心对象集合
    KeyObject = pd.DataFrame(columns = ["密度", "含糖率"])
    m = X.shape[0]  # 样本个数
    # 初始化分类标签结果
    Classid = np.zeros(m)
    Classid = Classid.tolist()
    for j in range(m):
        xj = X.loc[j]
        # 确定当前样本的E领域
        EArea, N_EArea = DetermineEArea(X,xj,epslion)
        if N_EArea >= MinPts:
            # 将样本添加到核心对象集合
            KeyObject = KeyObject.append(xj, ignore_index=False)  #忽略索引,往dataframe中插入一行数据
    # 初始化聚类簇、未访问样本集合
    k = 0
    Data_unused = X
    flag = KeyObject.empty
    while KeyObject.empty == False:
        # 记录当前未访问的样本集合
        Data_unused_old = Data_unused
        # 随机选取一个核心对象
        o_key = KeyObject.iloc[0]
        # 初始化Q集合
        Q_Object = pd.DataFrame(columns=["密度", "含糖率"])
        Q_Object = Q_Object.append(o_key, ignore_index=False)
        # 在Data_unused样本集中去除Q_Object中的核心对象
        Data_unused = DeleteSample(Data_unused, Q_Object)
        while Q_Object.empty == False:
            # 取出队列Q中的首个样本q
            q = Q_Object.iloc[0]
            # 确定q样本的E领域
            EArea_q, N_EArea_q = DetermineEArea(X, q, epslion)
            if N_EArea_q >= MinPts:
                Delta = pd.merge(Data_unused, EArea_q, on=["密度", "含糖率"], how='inner')
                # 将delta的样本并入队列Q
                Q_Object = pd.merge(Q_Object, Delta, how='outer')
                # 在Data样本集中去除Delta中的对象
                Data_unused = DeleteSample(Data_unused, Delta)
            # 判断q在Q_Object中的位置
            q = pd.DataFrame([q])
            # 在Q_Object样本集中去除q中的对象
            Q_Object = DeleteSample(Q_Object, q)
        k += 1
        # 在Data_unused_old样本集中去除Data_unused[取出Delta中的对象]
        Class = DeleteSample(Data_unused_old, Data_unused)
        # 在KeyObject样本集中去除Class中的对象[表示把已分类的样本取出来，继续分其他的样本]
        KeyObject = DeleteSample(KeyObject, Class)
        id = Class[:].index
        for i in id:
            Classid[i] = k

    return Classid,k

# main
if __name__ == '__main__':
    ## 生成数据集
    X = createData()
    ## 运行DBSCAN算法进行聚类
    Classid,K = DBSCAN(X, epslion = 0.11, MinPts = 5)
    print("类别：%d" %K)
    print("标签：%s" % Classid)
    ## 画出聚类结果
    Classid = np.array(Classid)
    for k in range(K+1):
        flag = Classid == k
        Xk = X[flag]
        plt.scatter(Xk["密度"],Xk["含糖率"],marker="o",s=80)
    plt.show()