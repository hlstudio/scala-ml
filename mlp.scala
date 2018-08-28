//:load d:/bigdata/ml/mlp.scala
:reset
:load d:/bigdata/ml/data.scala

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics._
/*
//XOR测试
val X=BDM(Array(0.0,0.0,0.0),Array(0.0,1.0,0.0),Array(1.0,0.0,0.0),Array(1.0,1.0,1.0))
val Y=BDM(Array(0.0,0.0),Array(1.0,0.0),Array(1.0,0.0),Array(0.0,0.0)) 
val layers=Array(3,5,2)    //三层，输入层3个特征，中间层8个特征，输出层2个特征

//样例测试
val X=BDM(Array(3.0,3.0),Array(4.0,3.0),Array(1.0,1.0),Array(1.0,2.0))
val Y=BDM(1.0,1.0,0.0,0.0)
val layers=Array(2,10,1)
*/
//iris
val layers=Array(4,10,3)
val (train,test)=iris
val (t1,t2)=oneBDM(train)
val X=t1
val Y=t2
//mnist
/*
val layers=Array(784,300,10)
val (t1,t2)=mnistBDM("train")
val X=t1
val Y=t2
*/

var W1=BDM.rand(layers(1),layers(0))*2.0 - 1.0d //第一层权重,W1和W2需要随机初始化
var B1=scala.util.Random.nextDouble    //第一层偏置
var H1=BDV.zeros[Double](layers(1)) //隐藏层节点
var W2=BDM.rand(layers(2),layers(1))*2.0 - 1.0d    //第二层权重
var B2=scala.util.Random.nextDouble    //第二层偏置
var O=BDV.zeros[Double](layers(2))    //输出层节点
val lr=0.6     //learnrate学习率
var E=1.0*Y.size    //总误差

//FP，前向计算结果
def fp(x:BDV[Double]){
     H1 = sigmoid(W1 * x +B1) //隐藏层计算，S激活
     O = sigmoid(W2 * H1 + B2) //输出层计算，S激活
}
//预测
def predict(x:BDV[Double]):BDV[Double]={fp(x);O}

//判断结果是否正确
def bingo(a:BDV[Double],b:BDV[Double]):Boolean={ argmax(a) == argmax(b)}

def fit(ir:Int){
    for(iter <- 1 to ir if E > 0.0001*Y.size){
        E = 0.0 
        var right =0.0
        val start=System.currentTimeMillis
        for(i <- 0 until X.rows){
            fp(X(i,::).t)
            if(bingo(O,Y(i,::).t)){right += 1.0}
            //BP，后向更新权重
            val T2=(((Y(i,::).t - O) :* O :* ( 1.0 - O)))    //计算theta2
            val T1=(T2.t * W2).t :* H1 :* (1.0 -H1)    //计算theta1
            W1 += lr * T1 * X(i,::)    //先更新W1，虽然反向计算但更新W1应该用W2的原值
            W2 += lr * T2 * H1.t //更新W2
            B1 += lr * sum(T1)    //更新B，TODO
            B2 += lr * sum(T2)
                
            E += sum(pow((Y(i,::).t - O),2) / O.size.toDouble)    //总误差
            //println(s"#$iter#$right#$E#${Y(i,::).t}#$O")
        }
        println(s"""#$iter#$right#${E.formatted("%.3f")}#${(1.0d*right/Y.rows).formatted("%.3f")}#${(System.currentTimeMillis - start)/1000}""")
    }
}

//测试集判断
def transform(t:(BDM[Double],BDM[Double])):Double = {
    var right=0.0
    for(i <- 0 until t._1.rows){
        fp(t._1(i,::).t)
        if(bingo(O,t._2(i,::).t)){right += 1.0}
        println(s"##$right##${t._2(i,::).t}##$O")
    }
    println(s"Accuracy test:${right/t._1.rows}")
    right/t._1.rows
}

fit(10)

//transform(mnistBDM("t10k"))
transform(oneBDM(test))