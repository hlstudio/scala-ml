//:load d:/bigdata/ml/bayes.scala
:reset
:load d:/bigdata/ml/data.scala

import scala.collection.immutable.Map
//求List[(Double, Int)]]中概率分布
def prop(list:List[(Double, Int)]):Map[Double,Double]={
    val mm=list.groupBy(_._1).mapValues(_.size)
    mm.toList.map(a =>(a._1,(1.0+a._2)/(1.0*mm.size+list.size))).toMap
}
def argmax(list:List[Double]):Int = list.zipWithIndex.maxBy(_._1)._2

val (train,test)=iris
//求各类别y的分布概率
val py=prop(train.map(a => (a._1,1)))
//y的类别数和x的特征数
val yk=py.size
val xk=train(0)._2.size

//总数
val cy=for(y <- 0 until yk) yield train.filter(_._1 == y).size
//求y=0/1/2,x1/x2/x3/x4的分布概率
val pyx=for(y <- 0 until yk ; x <- 0 until xk ) yield prop(train.filter(_._1 == y).flatMap(_._2.slice(x,x+1)).map((_,1)))

val all=for(p <- test;y <- 0 until yk ; x <- 0 until xk ) yield pyx(y*x+x).getOrElse(p._2(x),1.0/(cy(y)+1.0))
val tt=all.sliding(yk*xk,yk*xk).toList.map(_.sliding(xk,xk).toList).map(_.map(_.product))map(_.zipWithIndex.map(a => a._1 * py(a._2)))
val acc2=1.0d * tt.map(argmax(_)).zip(test.map(_._1.toInt)).filter(a => a._1 == a._2).size / test.size
println(s"Accuracy test:$acc2")
