//:load d:/bigdata/ml/mlp2.scala
:reset
:load d:/bigdata/ml/data.scala

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier

/*
val layers=Array(784,300,10)
val (t1,t2)=toDS(mnist)
//用小数据集试验
//val Array(train,test)=t2.randomSplit(Array(0.7,0.3))
val (train,test)=(t1,t2)
*/

//check iris
val layers=Array(4,8,3)
val (train,test)=toDS(iris)


val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128)
val model = trainer.fit(train)
val acc1=1.0d * model.transform(train).filter($"label" === $"prediction").count / train.count
val acc2=1.0d * model.transform(test).filter($"label" === $"prediction").count / test.count
println(s"Accuracy test:$acc2, train:$acc1")
