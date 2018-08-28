//:load d:/bigdata/ml/svm2.scala
:reset
:load d:/bigdata/ml/data.scala

import org.apache.spark.ml.classification.{LinearSVC,OneVsRest}

val (train,test)=toDS(mnist)

val svm = new LinearSVC()
svm.setRegParam(0.01).setMaxIter(30)
//二分类扩充为多分类
val ovr = new OneVsRest().setClassifier(svm)
val model = ovr.fit(train)

val acc1=1.0d * model.transform(train).filter($"label" === $"prediction").count / train.count
val acc2=1.0d * model.transform(test).filter($"label" === $"prediction").count / test.count
println(s"Accuracy test:$acc2, train:$acc1")