//:load d:/bigdata/ml/knn.scala
:reset
:load d:/bigdata/ml/data.scala
//ȡ���ڽ���k������
val k=9	
//����2������֮��ľ���
def dist(a:(Double,Array[Double]),b:(Double,Array[Double])):Double =  math.sqrt(a._2.zip(b._2).map(k =>math.pow(k._1 - k._2,2)).sum)
//ȡǰk�����������
def topk(ts:List[(Double,Array[Double])],s:(Double,Array[Double]),k:Int): List[Double] = ts.map(a => (a._1,dist(a,s))).sortBy(_._2).take(k).map(_._1)
//ȡk������������������
def most(ss:List[Double]):Double = ss.map((_,1)).groupBy(_._1).mapValues(_.size).toArray.sortBy(_._2).head._1

val (train,test)=iris
val acc=1.0d * test.map(a =>(a._1, most(topk(train,a,k)))).filter(a => a._1 == a._2).size /test.size

/*
val (train,test)=mnist
val t1=train.take(5000)
val t2=test.take(40)
val acc=1.0d * t2.map(a =>(a._1, most(topk(t1,a,k)))).filter(a => a._1 == a._2).size /t2.size
*/