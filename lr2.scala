//:load d:/bigdata/ml/lr2.scala
:reset
:load d:/bigdata/ml/data.scala

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val (train,test)=toDS(mnist)

val lr = new LogisticRegression()
/*
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.0001, 0.01, 1.0)).addGrid(lr.fitIntercept).addGrid(lr.maxIter, Array(10)).addGrid(lr.elasticNetParam, Array(0.1, 0.5, 1.0)).build()
 
// 80% of the data will be used for training and the remaining 20% for validation.  
val trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8)
 
// Run train validation split, and choose the best set of parameters.
val model = trainValidationSplit.fit(train)
model.bestModel.extractParamMap
*/
lr.setRegParam(0.0001).setElasticNetParam(1.0).setMaxIter(30)
val model = lr.fit(train)

val acc1=1.0d * model.transform(train).filter($"label" === $"prediction").count / train.count
val acc2=1.0d * model.transform(test).filter($"label" === $"prediction").count / test.count
println(s"Accuracy test:$acc2, train:$acc1")