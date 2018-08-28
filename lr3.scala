//:load d:/bigdata/ml/lr3.scala
:reset
:load d:/bigdata/ml/data.scala

//多分类逻辑回归
import breeze.linalg.{DenseVector => BDV}
import breeze.linalg._
import breeze.numerics._

/*
val (train,test)=iris
val (xtrain,ytrain)=oneBDM(train)
val (xtest,ytest)=oneBDM(test)
*/

val (xtrain,ytrain)=mnistBDM("train")
val (xtest,ytest)=mnistBDM("t10k")

var W = BDM.rand[Double](ytrain.cols,xtrain.cols)*2.0 - 1.0d  //随机初始化
var b = scala.util.Random.nextDouble
def multiLR(X:BDM[Double],Y:BDM[Double],epoch:Int = 10,learnRate:Double = 0.8){
        for(i <- 0 to epoch){
	       	var E = 0.0
	       	var right = 0.0 
            for(row <- 0 until X.rows){
                val xrow=X(row,::).t
                val O = sigmoid(W*xrow +b)
                if(argmax(O) == argmax(Y(i,::).t)){right += 1.0}
                val theta=(((Y(row,::).t - O) :* O :* ( 1.0 - O)))
                W += theta * xrow.t * learnRate
                b += learnRate * sum(theta)
                E += sum(pow((Y(row,::).t - O),2) / O.size.toDouble) 
           }
           println(s"#$i#$right#$E")
        }
} 
 
multiLR(xtrain,ytrain,50)

val acc1=1.0d * argmax(sigmoid(W*xtrain.t+b).apply(::,*)).t.toArray.zip(argmax(ytrain(*,::)).toArray).filter(a => a._1 == a._2).size / xtrain.rows
val acc2=1.0d * argmax(sigmoid(W*xtest.t+b).apply(::,*)).t.toArray.zip(argmax(ytest(*,::)).toArray).filter(a => a._1 == a._2).size / xtest.rows

println(s"Accuracy test:$acc2, train:$acc1")
