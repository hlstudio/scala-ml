//:load d:/bigdata/ml/mlp.scala
:reset
:load d:/bigdata/ml/data.scala

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics._
/*
//XOR����
val X=BDM(Array(0.0,0.0,0.0),Array(0.0,1.0,0.0),Array(1.0,0.0,0.0),Array(1.0,1.0,1.0))
val Y=BDM(Array(0.0,0.0),Array(1.0,0.0),Array(1.0,0.0),Array(0.0,0.0)) 
val layers=Array(3,5,2)    //���㣬�����3���������м��8�������������2������

//��������
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

var W1=BDM.rand(layers(1),layers(0))*2.0 - 1.0d //��һ��Ȩ��,W1��W2��Ҫ�����ʼ��
var B1=scala.util.Random.nextDouble    //��һ��ƫ��
var H1=BDV.zeros[Double](layers(1)) //���ز�ڵ�
var W2=BDM.rand(layers(2),layers(1))*2.0 - 1.0d    //�ڶ���Ȩ��
var B2=scala.util.Random.nextDouble    //�ڶ���ƫ��
var O=BDV.zeros[Double](layers(2))    //�����ڵ�
val lr=0.6     //learnrateѧϰ��
var E=1.0*Y.size    //�����

//FP��ǰ�������
def fp(x:BDV[Double]){
     H1 = sigmoid(W1 * x +B1) //���ز���㣬S����
     O = sigmoid(W2 * H1 + B2) //�������㣬S����
}
//Ԥ��
def predict(x:BDV[Double]):BDV[Double]={fp(x);O}

//�жϽ���Ƿ���ȷ
def bingo(a:BDV[Double],b:BDV[Double]):Boolean={ argmax(a) == argmax(b)}

def fit(ir:Int){
    for(iter <- 1 to ir if E > 0.0001*Y.size){
        E = 0.0 
        var right =0.0
        val start=System.currentTimeMillis
        for(i <- 0 until X.rows){
            fp(X(i,::).t)
            if(bingo(O,Y(i,::).t)){right += 1.0}
            //BP���������Ȩ��
            val T2=(((Y(i,::).t - O) :* O :* ( 1.0 - O)))    //����theta2
            val T1=(T2.t * W2).t :* H1 :* (1.0 -H1)    //����theta1
            W1 += lr * T1 * X(i,::)    //�ȸ���W1����Ȼ������㵫����W1Ӧ����W2��ԭֵ
            W2 += lr * T2 * H1.t //����W2
            B1 += lr * sum(T1)    //����B��TODO
            B2 += lr * sum(T2)
                
            E += sum(pow((Y(i,::).t - O),2) / O.size.toDouble)    //�����
            //println(s"#$iter#$right#$E#${Y(i,::).t}#$O")
        }
        println(s"""#$iter#$right#${E.formatted("%.3f")}#${(1.0d*right/Y.rows).formatted("%.3f")}#${(System.currentTimeMillis - start)/1000}""")
    }
}

//���Լ��ж�
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