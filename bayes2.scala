//:load d:/bigdata/ml/bayes2.scala
:reset
:load d:/bigdata/ml/data.scala

import org.apache.spark.ml.classification.NaiveBayes
val (train,test)=toDS(mnist)
val model =new NaiveBayes().fit(train)
val acc1=1.0d * model.transform(train).filter($"label" === $"prediction").count / train.count
val acc2=1.0d * model.transform(test).filter($"label" === $"prediction").count / test.count
println(s"Accuracy test:$acc2, train:$acc1")