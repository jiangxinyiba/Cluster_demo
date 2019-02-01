# Cluster_demo
A python demo which consist of some cluster algorithm

# DBSCAN算法
根据西瓜书第213页伪代码实现的DBSCAN聚类算法

# Logistic回归算法
根据机器学习实战第5章“Logistic回归”，基于附属的代码修改后可以在python3下正常运行，
修改说明如下：
1.部分代码按照python3的逻辑进行修改
2.在colicTest函数中，基于当前数据的各列特征之间范围差距过大的问题，导致math.exp()无法求解，对训练数据进行归一化处理，其中归一化采用scikit中的preprocess库实现。
3.各梯度算法函数说明。
  gradAscent是梯度上升算法；
  stocGradAscent0是普通的随机梯度上升算法，采用增量式更新权值，训练了所有的训练数据一遍；
  stocGradAscent1是改进的随机梯度上升算法，进行了numIter次迭代，每次都是随机选取样本增量式更新权值,本函数用于马疝分类问题；
  stocGradAscent1_gai改进的随机梯度上升算法，修改后可以展示每个权值的更新情况，本函数用于testSet.txt的数据分类展示。
