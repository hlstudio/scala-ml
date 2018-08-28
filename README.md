李航《统计学习方法》的scala版本实现

开发环境
spark-2.3.1

分类方法准确率对比
||Iris|Mnist|Mnist备注|Iris|Mnist|Mnist备注|
|----------|----------|----------|--------------------|----------|----------|--------------------||
|多层感知机|1.00|0.97|10Epoch，20分钟|0.97|0.96/0.98|784-300-10，5分钟|
|KNN分类|0.80|0.65|抽样，快|||
|Bayes分类|0.43||OOM|0.94|0.83/0.82|快|
|决策树分类|0.80|0.79|抽样，仍慢|1.00|0.68/0.68|快|
|逻辑回归|0.97|0.99|均简化二分类|1.00/0.98|0.92/0.93|先交叉验证找超参，慢|
|SVM|1.00||简化二分类|0.93/0.84|0.91/0.91|OneVsRest|

每行前一组为自己的实现的算法，后一组为spark自带的算法
双值前面为测试集准确率，后者为训练集准确率
