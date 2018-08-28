//:load d:/bigdata/ml/lr.scala
:reset
:load d:/bigdata/ml/data.scala

import breeze.linalg.{DenseVector => BDV}
import breeze.linalg._
import breeze.numerics._

def norm(label:Double):Double= if(label != 0.0 ) 1.0 else 0.0  //将多类转化为二分类

val List(train,test)=iris.productIterator.map(_.asInstanceOf[List[LabeledArray]].map(a => (norm(a._1),a._2))).toList
val (x1,_)=oneBDM(train)
val X=x1
val y=train.map(_._1)
val (x2,_)=oneBDM(test)
val Xt=x2
val yt=test.map(_._1)

/*
val (x1,y1)=mnistBDM("train")
val X=x1
val y=argmax(y1(*,::)).toArray.map(norm(_)).toList
val (x2,y2)=mnistBDM("t10k")
val Xt=x2
val y=argmax(y2(*,::)).toArray.map(norm(_)).toList
*/

def binLR(X:BDM[Double],y:List[Double],epoch:Int = 10,learnRate:Double = 0.8):(BDV[Double],Double)={
        var W = BDV.rand[Double](X.cols)*2.0 - 1.0d  //随机初始化
        var b = 0.0
        println("row\t output\t w1\t w2\t b")
        for(i <- 0 to epoch){
                for(row <- 0 until X.rows){
                    val xrow=X(row,::).t
                    val output = sigmoid(xrow.dot(W) +b)
                    val theta=(y(row)-output)*output*(1-output)
                    W += xrow * learnRate * theta
                    b += learnRate * theta
                    println(s"$row\t $output\t ${W(0)}\t ${W(1)}\t $b")
               }
        }
        (W,b)
}  
val (w,b)=binLR(X,y,20)

val acc1=1.0d * sigmoid(X*w+b).toArray.zip(y).filter(a => abs(a._1 - a._2) < 0.1).size / X.rows
val acc2=1.0d * sigmoid(Xt*w+b).toArray.zip(yt).filter(a => abs(a._1 - a._2) < 0.1).size / Xt.rows
println(s"Accuracy test:$acc2, train:$acc1")
