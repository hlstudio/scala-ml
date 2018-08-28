//:load d:/bigdata/ml/data.scala
import scala.io.Source
import java.io._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql.Dataset
import breeze.linalg.{DenseMatrix => BDM}
//��������Լ���ĸ�ʽΪ(Double, Array[Double]),ǰ��Ϊlabel������Ϊfeatures
//���ݼ��ķ��ظ�ʽΪ(train,test)������train��test����������list
type LabeledArray= (Double, Array[Double])

//ת��ΪBDM
def oneBDM(t:List[LabeledArray]):(BDM[Double],BDM[Double])={
    val (xd,yd)=(t.map(_._2),t.map(_._1))
    val X=BDM.tabulate(xd.size,xd(0).size){case (i, j) => xd(i)(j)}
    val Y=BDM.zeros[Double](yd.size,yd.max.toInt+1)
    yd.zipWithIndex.foreach(a => Y(a._2,a._1.toInt) = 1.0)
    (X,Y)    
}
//ת��Ϊdataset
def oneDS(t:List[LabeledArray]):Dataset[LabeledPoint] = sc.makeRDD(t.map(a => LabeledPoint(a._1,Vectors.dense(a._2)))).toDS
def toDS(tt:(List[LabeledArray],List[LabeledArray])):(Dataset[LabeledPoint],Dataset[LabeledPoint])= (oneDS(tt._1),oneDS(tt._2)) 

//iris���ݼ�,�̶�train��test
def iris():(List[LabeledArray],List[LabeledArray])={
    val fs=Source.fromFile("d:/bigdata/ml/data/iris.data").getLines.map(_.split(",")).toList
    //string to index
    val ymap=fs.map(_.last).distinct.zipWithIndex.map(a =>(a._1,a._2.toDouble)).toMap
    val data=fs.map(a =>(ymap(a.last),a.slice(0,4).map(_.toDouble)))
    //˳����3�࣬ÿ50��һ��
    val train=util.Random.shuffle(data.slice(0,40) ++ data.slice(50,90) ++ data.slice(100,140))
    val test=util.Random.shuffle(data.slice(40,50) ++ data.slice(90,100) ++ data.slice(140,150))
    //ǰ40Ϊtrain����10Ϊtest
    (train,test)
}

//mnist���ݼ�
def readImage(fname:String):Array[Int]={
    val file=new DataInputStream(new BufferedInputStream(new FileInputStream(fname)))
    val magic=file.readInt
    val num=file.readInt
    val rows=file.readInt
    val cols=file.readInt
    val data=for(a <- 0 until num*rows*cols) yield (file.readByte)
    file.close()
    data.map(_ & 0xFF).toArray    //�����ֽ����Ϊ��
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
//oneBDMת��mnist���ݼ������������Ż���
def mnistBDM(name:String):(BDM[Double],BDM[Double])={
    val xd=readImage(s"d:/bigdata/ml/data/${name}-images.idx3-ubyte").map(_.toDouble/255)
    val yd=readLabel(s"d:/bigdata/ml/data/${name}-labels.idx1-ubyte").map(_.toDouble)
    val cols=xd.size/yd.size
	val X=BDM.tabulate(yd.size,cols){case (i, j) => xd(i*cols + j)}
	val Y=BDM.zeros[Double](yd.size,10)
	yd.zipWithIndex.foreach(a => Y(a._2,a._1.toInt) = 1.0)
	(X,Y)	
}
	