//:load d:/bigdata/ml/cart2.scala
:reset
:load d:/bigdata/ml/data.scala

import org.apache.spark.ml.classification.DecisionTreeClassifier

val t1=List((0.0,Array(1.0 ,2.0 ,2.0 ,3.0 )),(0.0,Array(1.0 ,2.0 ,2.0 ,2.0 )),(1.0,Array(1.0 ,1.0 ,2.0 ,2.0 )),(1.0,Array(1.0 ,1.0 ,1.0 ,3.0 )),(0.0,Array(1.0 ,2.0 ,2.0 ,3.0 )),(0.0,Array(2.0 ,2.0 ,2.0 ,3.0 )),(0.0,Array(2.0 ,2.0 ,2.0 ,2.0 )),(1.0,Array(2.0 ,1.0 ,1.0 ,2.0 )),(1.0,Array(2.0 ,2.0 ,1.0 ,1.0 )),(1.0,Array(2.0 ,2.0 ,1.0 ,1.0 )),(1.0,Array(3.0 ,2.0 ,1.0 ,1.0 )),(1.0,Array(3.0 ,2.0 ,1.0 ,2.0 )),(1.0,Array(3.0 ,1.0 ,2.0 ,2.0 )),(1.0,Array(3.0 ,1.0 ,2.0 ,1.0 )),(0.0,Array(3.0 ,2.0 ,2.0 ,3.0 )))
val (train,test)=(oneDS(t1),oneDS(t1))

val (t1,test)=toDS(iris)
val Array(train,dev)=t1.randomSplit(Array(0.75,0.25))

val model =new DecisionTreeClassifier().setImpurity("gini").fit(train)
val acc1=1.0d * model.transform(train).filter($"label" === $"prediction").count / train.count
val acc2=1.0d * model.transform(test).filter($"label" === $"prediction").count / test.count
println(s"Accuracy test:$acc2, train:$acc1")
model.toDebugString