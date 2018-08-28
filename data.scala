//:load d:/bigdata/ml/data.scala
import scala.io.Source
import java.io._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql.Dataset
import breeze.linalg.{DenseMatrix => BDM}
//单个样本约定的格式为(Double, Array[Double]),前面为label，后面为features
//数据集的返回格式为(train,test)，其中train和test都是样本的list
type LabeledArray= (Double, Array[Double])

//转换为BDM
def oneBDM(t:List[LabeledArray]):(BDM[Double],BDM[Double])={
    val (xd,yd)=(t.map(_._2),t.map(_._1))
    val X=BDM.tabulate(xd.size,xd(0).size){case (i, j) => xd(i)(j)}
    val Y=BDM.zeros[Double](yd.size,yd.max.toInt+1)
    yd.zipWithIndex.foreach(a => Y(a._2,a._1.toInt) = 1.0)
    (X,Y)    
}
//转换为dataset
def oneDS(t:List[LabeledArray]):Dataset[LabeledPoint] = sc.makeRDD(t.map(a => LabeledPoint(a._1,Vectors.dense(a._2)))).toDS
def toDS(tt:(List[LabeledArray],List[LabeledArray])):(Dataset[LabeledPoint],Dataset[LabeledPoint])= (oneDS(tt._1),oneDS(tt._2)) 

//iris数据集,固定train和test
def iris():(List[LabeledArray],List[LabeledArray])={
    val fs=Source.fromFile("d:/bigdata/ml/data/iris.data").getLines.map(_.split(",")).toList
    //string to index
    val ymap=fs.map(_.last).distinct.zipWithIndex.map(a =>(a._1,a._2.toDouble)).toMap
    val data=fs.map(a =>(ymap(a.last),a.slice(0,4).map(_.toDouble)))
    //顺序存放3类，每50个一类
    val train=util.Random.shuffle(data.slice(0,40) ++ data.slice(50,90) ++ data.slice(100,140))
    val test=util.Random.shuffle(data.slice(40,50) ++ data.slice(90,100) ++ data.slice(140,150))
    //前40为train，后10为test
    (train,test)
}

//mnist数据集
def readImage(fname:String):Array[Int]={
    val file=new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic=file.readInt
    val num=file.readInt
    val rows=file.readInt
    val cols=file.readInt
    val data=for(a <- 0 until num*rows*cols) yield (file.readByte)
    file.close()
    data.map(_ & 0xFF).toArray    //处理字节溢出为负
}
def readLabel(fname:String):Array[Byte]={
    val file=new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic=file.readInt
    val num=file.readInt
    val data=for(a <- 0 until num) yield (file.readByte)
    file.close()
    data.toArray
}
def loadData(name:String):List[LabeledArray]={
    val xd=readImage(s"d:/bigdata/ml/data/${name}-images.idx3-ubyte").map(_.toDouble)
    val yd=readLabel(s"d:/bigdata/ml/data/${name}-labels.idx1-ubyte").map(_.toDouble)
    val cols=xd.size/yd.size
    //val data=Array.tabulate(yd.size,cols){case (i, j) => xd(cols*i + j)}
    val data=xd.sliding(cols,cols).toArray
    yd.zip(data).toList
}
def mnist():(List[LabeledArray],List[LabeledArray])=(loadData("train"),loadData("t10k"))
//oneBDM转换mnist数据集超慢，单独优化下
def mnistBDM(name:String):(BDM[Double],BDM[Double])={
    val xd=readImage(s"d:/bigdata/ml/data/${name}-images.idx3-ubyte").map(_.toDouble/255)
    val yd=readLabel(s"d:/bigdata/ml/data/${name}-labels.idx1-ubyte").map(_.toDouble)
    val cols=xd.size/yd.size
	val X=BDM.tabulate(yd.size,cols){case (i, j) => xd(i*cols + j)}
	val Y=BDM.zeros[Double](yd.size,10)
	yd.zipWithIndex.foreach(a => Y(a._2,a._1.toInt) = 1.0)
	(X,Y)	
}
	