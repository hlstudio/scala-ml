//:load d:/bigdata/ml/svm.scala
:reset
:load d:/bigdata/ml/data.scala

import scala.collection.mutable.ArrayBuffer
import scala.math.signum
import breeze.linalg._

def norm(label:Double):Double= if(label != 0.0 ) 1.0 else -1.0  //������ת��Ϊ������

//֧����������
case class SVM(alpha: Array[Double], //֧��������Ӧ�ĳ���
               y: Array[Double], //֧��������Ӧ�ı�ǩ
               sv: Array[Array[Double]], //֧������
               b: Double,
               kel: (Array[Double], Array[Double]) => Double) {

  def predict(x: Array[Double]) = { //Ԥ��һ����, ���ϵĹ�ʽ
    var sum = b
    alpha.indices.foreach {
      i => sum += alpha(i) * y(i) * kel(x, sv(i))
    }
    signum(sum)
  }
}

//ר�ŷź˺����Ķ���
object Kennels {
  def innerProduct(x: Array[Double], y: Array[Double]): Double = sum(x * y)
}

//���ѡ��ڶ�������
def selectJ(i: Int, m: Int): Int = { 
  var j = i
  while (j == i) {
    j = util.Random.nextInt(m)
  }
  j
}
//�ü�alpha2 �����㲻��ʽԼ��
def clipAlpha(aj: Double, H: Double, L: Double): Double = if (aj > H) H else if (aj < L) L else aj

def simpleSMO(dataMat: Array[Array[Double]], y: Array[Double], C: Double, kel: (Array[Double], Array[Double]) => Double, toler: Double, maxIter: Int): SVM = {
  val m = dataMat.length
  var alphaPairChanged, iter = 0
  var fXi, fXj, eta, Ei, Ej, alphaIold, alphaJold, L, H, b, b1, b2 = 0.0
  val alpha = new Array[Double](m)

  //����һ�����g(x_i), �������p127
  def g(n: Int) = {
    var sum = b
    (0 until m).foreach(k => sum += alpha(k) * y(k) * kel(dataMat(n), dataMat(k)))
    sum
  }

  while (iter < maxIter) {
    alphaPairChanged = 0
    for (i <- 0 until m) {
      fXi = g(i)
      Ei = fXi - y(i) //����Ei
      if ((Ei * y(i) < -toler && alpha(i) < C) || (Ei * y(i) > toler && alpha(i) > 0)) {
        val j = selectJ(i, m)
        fXj = g(j)
        Ej = fXj - y(j)
        alphaIold = alpha(i)
        alphaJold = alpha(j)
        // ����L, ��H ,�μ���ع�ʽ, ��Ҫ��i=j��i!=j�������
        if (y(i) != y(j)) {
          L = 0.0.max(alphaJold - alphaIold)
          H = C.min(C + alphaJold - alphaIold)
        }
        else {
          L = 0.0.max(alphaJold + alphaIold - C)
          H = C.min(alphaJold + alphaIold)
        }
        if (L == H) println("L == H, ��һ��")
        else {
          // eta =  K_11 + K_22 - 2*K_12, ע�������python��Ĳ�һ��, ��һ�����ŵ�����, ���ǲ�������ϵĹ�ʽ
          eta = kel(dataMat(i), dataMat(i)) + kel(dataMat(j), dataMat(j)) - 2 * kel(dataMat(i), dataMat(j))
          if (eta < 0.00001) println("eta ̫С��Ϊ��, ��һ��")
          else {
            alpha(j) = alphaJold + y(j) * (Ei - Ej) / eta
            alpha(j) = clipAlpha(alpha(j), H, L)
            if (math.abs(alpha(j) - alphaJold) < 0.00001) println("j not moving enough") // alphaJ �޸���̫С������
            else {
	          //����alphaI alpha_i���޸ķ����෴
              alpha(i) = alphaIold + y(i) * y(j) * (alphaJold - alpha(j))
              // ����b,Ϊ����alpha���ó�����b
              b1 = b - Ei - y(i) * kel(dataMat(i), dataMat(i)) * (alpha(i) - alphaIold)
                  -y(j) * kel(dataMat(j), dataMat(i)) * (alpha(j) - alphaJold)
              b2 = b - Ej - y(i) * kel(dataMat(i), dataMat(j)) * (alpha(i) - alphaIold)
                  -y(j) * kel(dataMat(j), dataMat(j)) * (alpha(j) - alphaJold)
              if (alpha(i) > 0 && alpha(i) < C) b = b1
              else if (alpha(j) > 0 && alpha(j) < C) b = b2
              else b = (b1 + b2) / 2.0
              alphaPairChanged += 1
              println("iter %d, alpha changed %d".format(iter, alphaPairChanged))
            }
          }
        }
      }
    }
    if (alphaPairChanged == 0) iter += 1 else iter = 0
    println("iteration number: %d".format(iter))
  }
  val svIndex = alpha.indices.filter(i => alpha(i) > 0).toArray // ��ѡ֧�������ı��
  val supportVectors = svIndex.map(i => dataMat(i)) //��ѡ֧������
  val svY = svIndex.map(i => y(i)) //��ѡ��ǩ
  new SVM(svIndex.map(i => alpha(i)), svY, supportVectors, b, kel) //����һ��֧��������
}


val List(train,test)=iris.productIterator.map(_.asInstanceOf[List[LabeledArray]].map(a => (norm(a._1),a._2))).toList
val dataMat=train.map(_._2).toArray
val y=train.map(_._1).toArray

val tX=test.map(_._2).toArray
val tY=test.map(_._1).toArray

val kel =  Kennels.innerProduct _ //����˺���
val svm1 = simpleSMO(dataMat, y, 0.7, kel, 0.0005, 40) //ѵ��

val acc1=1.0d * dataMat.map(svm1.predict(_)).zip(y).filter(a => a._1 == a._2).size /y.size
val acc2=1.0d * tX.map(svm1.predict(_)).zip(tY).filter(a => a._1 == a._2).size /tY.size
println(s"Accuracy test:$acc2, train:$acc1")