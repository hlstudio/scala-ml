//:load d:/bigdata/ml/hmm.scala
import scala.collection.mutable.ArrayBuffer
import java.io._
import org.apache.spark.rdd._
import scala.reflect.ClassTag

//定义hmm model states 隐状态  start_p 初始概率（隐状态） trans_p 转移概率（隐状态） emit_p 发射概率 （隐状态表现为显状态的概率）
case class HmmModel(var states:Array[Int],var chars:Array[Char],var start_p:Array[Double],var trans_p:Array[Array[Double]],var emit_p:Array[Array[Double]]){
    //求解HMM模型 @see https://github.com/hankcs/Viterbi/blob/master/src/com/hankcs/algorithm/Viterbi.java 
    def viterbi(obs:Array[Int]):Array[Int]={
        val V = Array.ofDim[Double](obs.length,states.length)
        var path = Array.ofDim[Int](states.length,obs.length)
        for (y <- states){
            V(0)(y) = start_p(y) * emit_p(y)(obs(0))
            path(y)(0) = y
        }
        
        for (t <- 1 until obs.length){
            val newpath = Array.ofDim[Int](states.length,obs.length)
            for (y <- states){
                val (state,prob)={for(y0 <- states) yield (y0,V(t - 1)(y0) * trans_p(y0)(y) * emit_p(y)(obs(t)))}.maxBy(_._2)
                V(t)(y) = prob
                System.arraycopy(path(state), 0, newpath(y), 0, t)
                newpath(y)(t) = y
            }
            path = newpath
        }
        val (state,prob)={for(y <- states) yield(y,V(obs.length - 1)(y))}.maxBy(_._2)
        path(state)
    }
    
    //依据BMES生成分词结果
    def eject(list:String,bmes_a:Array[Char]):List[String] ={
        val ab=new ArrayBuffer[String]
        var start = -1;
        for(step <- 0 until bmes_a.size){
            bmes_a(step) match {
                case 'B' => {start = step}
                case 'S' => {ab += list.slice(step,step+1).mkString;start = -1}
                case 'E' => {ab += list.slice(start,step+1).mkString;start = -1}
                case _ => Nil
            }
        }
        ab.toList
    }    
    
    //采用hmm模型分词
    def seg(str:String):List[String]={
        val bmes=Array('B','M','E','S')
        val result=viterbi(str.map(chars.indexOf(_)).toArray).map(bmes(_))
        eject(str,result)
    }

    //保存模型，TODO:不应该写入整个对象，应该只写入纯数据部分，否则修改方法会引起序列化的版本差异造成load出错
    def save(path:String){
        val oos = new ObjectOutputStream(new FileOutputStream(path))
        oos.writeObject(this)
        oos.close()        
    }
}

//加载模型
def load(path:String):HmmModel={
    val ois = new ObjectInputStream(new FileInputStream(path))
    val h=ois.readObject.asInstanceOf[HmmModel]
    ois.close
    h
}

//N-GRAM元分词
def ngram(line:String,n:Int):Map[String,Int] = { 
    line.toCharArray.sliding(n,1).map(a => (a.mkString,1L)).filter(_._1.length == n).toList.groupBy(_._1).mapValues(_.size)
}

//zipShuffle
implicit class RichContext[T](rdd: RDD[T]) {
  def zipShuffle[A](other: RDD[A])(implicit kt: ClassTag[T], vt: ClassTag[A]): RDD[(T, A)] = {
    val otherKeyd: RDD[(Long, A)] = other.zipWithIndex().map { case (n, i) => i -> n }
    val thisKeyed: RDD[(Long, T)] = rdd.zipWithIndex().map { case (n, i) => i -> n }
    val joined                    = new PairRDDFunctions(thisKeyed).join(otherKeyd).map(_._2)
    joined
  }
}

//抽取的公用方法，更新概率数组
def updateArray(dic:Array[(Char, Int)],chars:Array[Char],rowIndex:Int,array:Array[Array[Double]]){
    val rowTotal=dic.map(_._2).sum
    dic.foreach(column =>{
        val columnIndex=chars.indexOf(column._1)
        val ratio=1.0d*column._2 / rowTotal
        array(rowIndex)(columnIndex) = ratio
    })    
}

//依据标注语料生成hmm模型,file1为中文，file2为BMES标注
def train(file1:String,file2:String):HmmModel={
    val c1=sc.textFile(file1)
    val c2=sc.textFile(file2)
    //总长度
    val total=c1.map(_.size).sum
    //观测集长度
    val chars=c1.flatMap(_.toString).distinct.collect.sorted
    //计算π初始概率
    val pi=c2.flatMap(_.toString).map((_,1)).reduceByKey(_+_).map(a =>(a._1,1.0d*a._2/total)).collect
    //顺序为BMES
    val bmes=Array('B','M','E','S')
    val states=Array(0,1,2,3)
    //初始概率pi
    val start_p=pi.sorted.map(_._2)
    val temp=start_p(2);start_p(2)=start_p(1);start_p(1)=temp
	
    //bmes的转移概率A
    val trans=c2.flatMap(a => ngram(a,2)).reduceByKey(_+_)
    val trans_p=Array.ofDim[Double](states.length,states.length)
    for(row <- bmes){
        val rowIndex=bmes.indexOf(row)
        val dic=trans.filter(_._1(0) == row).map(a =>(a._1(1),a._2)).reduceByKey(_+_).collect
        updateArray(dic,bmes,rowIndex,trans_p)
    }

    //发射概率B,先粗粒度zipShuffle再zip提升性能几十倍
    val c1c2=c1.zipShuffle(c2)
    val emit=c1c2.flatMap(a => a._1.zip(a._2))
    val emit_p=Array.ofDim[Double](states.length,chars.length)
    for(row <- bmes){
        val rowIndex=bmes.indexOf(row)
        val dic=emit.filter(_._2 == row).map(a =>(a._1,1)).reduceByKey(_+_).collect
        updateArray(dic,chars,rowIndex,emit_p)
    }
    
    //至此，HMM的参数都求解了
    HmmModel(states, chars, start_p, trans_p,emit_p)
}

val hmm=train("d:/bigdata/ml/data/corpus1c.txt","d:/bigdata/ml/data/corpus2c.txt")
val test=Array("文化和旅游发展服务满足游客的体验需求拉动城市经济",
"对模型参数进行训练在分词阶段再通过模型计算各种分词出现的概率",
"龚学平等表示会保证金云鹏的安全",
"严守一把手机关了",
"和贾冰一个单位的杜丽丽已经和她的男朋友武惠良在贾冰家帮他老婆准备这顿饭了",
"深夜的穆赫兰道发生一桩车祸女子丽塔在车祸中失忆了",
"公交车中将不允许吃东西",
"蒂莫西伊斯顿在伦敦的金斯敦公司上班",
"舒适性胎噪风噪偏大避震偏硬过坎弹跳明显",
"本季最强家庭瘦腰计划彻底告别大肚腩没有腹肌的人生是不完整的")
test.map(hmm.seg(_)).foreach(println)
