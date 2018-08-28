//:load d:/bigdata/ml/svm.scala
:reset
:load d:/bigdata/ml/data.scala

import scala.collection.mutable.ArrayBuffer
import scala.math.signum
import breeze.linalg._

def norm(label:Double):Double= if(label != 0.0 ) 1.0 else -1.0  //将多类转化为二分类

//支持向量的类
case class SVM(alpha: Array[Double], //支持向量对应的乘子
               y: Array[Double], //支持向量对应的标签
               sv: Array[Array[Double]], //支持向量
               b: Double,
               kel: (Array[Double], Array[Double]) => Double) {

  def predict(x: Array[Double]) = { //预测一个点, 书上的公式
    var sum = b
    alpha.indices.foreach {
      i => sum += alpha(i) * y(i) * kel(x, sv(i))
    }
    signum(sum)
  }
}

//专门放核函数的对象
object Kennels {
  def innerProduct(x: Array[Double], y: Array[Double]): Double = sum(x * y)
}

//随机选择第二个变量
def selectJ(i: Int, m: Int): Int = { 
  var j = i
  while (j == i) {
    j = util.Random.nextInt(m)
  }
  j
}
//裁剪alpha2 以满足不等式约束
def clipAlpha(aj: Double, H: Double, L: Double): Double = if (aj > H) H else if (aj < L) L else aj

def simpleSMO(dataMat: Array[Array[Double]], y: Array[Double], C: Double, kel: (Array[Double], Array[Double]) => Double, toler: Double, maxIter: Int): SVM = {
  val m = dataMat.length
  var alphaPairChanged, iter = 0
  var fXi, fXj, eta, Ei, Ej, alphaIold, alphaJold, L, H, b, b1, b2 = 0.0
  val alpha = new Array[Double](m)

  //定义一个输出g(x_i), 见李航书上p127
  def g(n: Int) = {
    var sum = b
    (0 until m).foreach(k => sum += alpha(k) * y(k) * kel(dataMat(n), dataMat(k)))
    sum
  }

  while (iter < maxIter) {
    alphaPairChanged = 0
    for (i <- 0 until m) {
      fXi = g(i)
      Ei = fXi - y(i) //计算Ei
      if ((Ei * y(i) < -toler && alpha(i) < C) || (Ei * y(i) > toler && alpha(i) > 0)) {
        val j = selectJ(i, m)
        fXj = g(j)
        Ej = fXj - y(j)
        alphaIold = alpha(i)
        alphaJold = alpha(j)
        // 计算L, 和H ,参见相关公式, 主要分i=j和i!=j两种情况
        if (y(i) != y(j)) {
          L = 0.0.max(alphaJold - alphaIold)
          H = C.min(C + alphaJold - alphaIold)
        }
        else {
          L = 0.0.max(alphaJold + alphaIold - C)
          H = C.min(alphaJold + alphaIold)
        }
        if (L == H) println("L == H, 下一轮")
        else {
          // eta =  K_11 + K_22 - 2*K_12, 注意这里和python版的不一样, 有一个负号的区别, 这是采用李航书上的公式
          eta = kel(dataMat(i), dataMat(i)) + kel(dataMat(j), dataMat(j)) - 2 * kel(dataMat(i), dataMat(j))
          if (eta < 0.00001) println("eta 太小或为负, 下一轮")
          else {
            alpha(j) = alphaJold + y(j) * (Ei - Ej) / eta
            alpha(j) = clipAlpha(alpha(j), H, L)
            if (math.abs(alpha(j) - alphaJold) < 0.00001) println("j not moving enough") // alphaJ 修改量太小就跳过
            else {
	          //更新alphaI alpha_i的修改方向相反
              alpha(i) = alphaIold + y(i) * y(j) * (alphaJold - alpha(j))
              // 更新b,为两个alpha设置常数项b
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
  val svIndex = alpha.indices.filter(i => alpha(i) > 0).toArray // 挑选支持向量的编号
  val supportVectors = svIndex.map(i => dataMat(i)) //挑选支持向量
  val svY = svIndex.map(i => y(i)) //挑选标签
  new SVM(svIndex.map(i => alpha(i)), svY, supportVectors, b, kel) //创建一个支持向量机
}


val List(train,test)=iris.productIterator.map(_.asInstanceOf[List[LabeledArray]].map(a => (norm(a._1),a._2))).toList
val dataMat=train.map(_._2).toArray
val y=train.map(_._1).toArray

val tX=test.map(_._2).toArray
val tY=test.map(_._1).toArray

val kel =  Kennels.innerProduct _ //导入核函数
val svm1 = simpleSMO(dataMat, y, 0.7, kel, 0.0005, 40) //训练

val acc1=1.0d * dataMat.map(svm1.predict(_)).zip(y).filter(a => a._1 == a._2).size /y.size
val acc2=1.0d * tX.map(svm1.predict(_)).zip(tY).filter(a => a._1 == a._2).size /tY.size
println(s"Accuracy test:$acc2, train:$acc1")