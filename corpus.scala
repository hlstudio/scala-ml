//:load d:/bigdata/ml/corpus.scala
import org.apache.spark.rdd.RDD
import scala.util.Random
import scala.collection.mutable.ArrayBuffer

val homeDir= "c:/temp/raw/data"
//处理词典
def transDict():RDD[(String,Int)]={
    val han=sc.textFile(s"${homeDir}/hanlp/*.txt")
    //选择十元词及以下
    val one=han.map(_.split("[ \t]")).filter(a => !a(0).contains("##") && a(0).length < 11 && a.length < 8).map(a =>a match {
        case Array(x1,y1,z1) => (x1,y1,z1.toInt)
        case Array(x1,y1,z1,y2,z2) => (x1,y1,z1.toInt+z2.toInt)
        case Array(x1,y1,z1,y2,z2,y3,z3) => (x1,y1,z1.toInt+z2.toInt+z3.toInt)
    })
    //跳过词性，去重
    one.map(a =>(a._1,a._3)).distinct
}

//词序列标注
def bmes(key:String):String={
    key.length match {
        case 1 => "S"
        case 2 => "BE"
        case _ => "B"+List().padTo(key.length -2,"M").mkString+"E"
    }
}

//权重数组
def weightArray():Array[Int]={
    val a=new Array[Int](1000)
    a(0)=0
    for( i <- 1 until 1000) a(i)=a(i-1)+i
    a
}

//二分查找
def binarySearch[A <% Ordered[A]](a: IndexedSeq[A], v: A) :Int = {
  def recurse(low: Int, high: Int): Int = (low + high) / 2 match {
    case _ if high < low => high
    case mid if a(mid) > v => recurse(low, mid - 1)
    case mid if a(mid) < v => recurse(mid + 1, high)
    case mid => mid
  }
  recurse(0, a.size - 1)
}

//随机取词
def randWord(cand:Map[Int,List[String]],index:Int):String={
    //取对应权重组
    val group=cand(index+1)
    //然后随机取一个词
    group(Random.nextInt(group.size))
}

//权重抽词，len为抽词数量
def makeCorpus(len:Int):(RDD[(String,String)])={
    val wa=weightArray
    val one=transDict
    val cand=one.map(a => (a._1,if (a._2 > 1000) 1000 else a._2)).groupBy(_._2).map(a=>(a._1,a._2.toList.map(_._1))).collectAsMap.toMap
    val mm=(1 to 1000).sum
    //随机抽取
    val r1=sc.makeRDD(1 to len).map(a => Random.nextInt(mm)).map(a => binarySearch(wa,a)).map(a => randWord(cand,a))
    //原始词全部随机附加
    val r2=sc.makeRDD(one.map(_._1).takeSample(false,one.count.toInt))
    val r=r1 ++ r2
    r.map(a =>(a,bmes(a)))
}
def log(filename:String,s:String){scala.tools.nsc.io.File(filename).appendAll(s+"\n")}
//保存语料
def saveCorpus(c:RDD[(String,String)]){
    //平均长度为3.2，则每8个词生成一个句子来保存
    val file1=s"${homeDir}/hanlp/corpus1t.txt"
    val file2=s"${homeDir}/hanlp/corpus2t.txt"
    val ab=new ArrayBuffer[(String,String)]
    for(one <- c.zipWithIndex){
        ab += one._1
        if(one._2 >0 && one._2 % 8 == 0){
            log(file1,ab.map(_._1).mkString)
            log(file2,ab.map(_._2).mkString)
            ab.clear
        }
    }
    if(!ab.isEmpty){
        log(file1,ab.map(_._1).mkString)
        log(file2,ab.map(_._2).mkString)
    }
}

//依据带词频词库生成语料
//saveCorpus(makeCorpus(80000*10000))

//转换语料，将形如 雷达/n 找/v 不到/v 他/r 。/w  转换为corpus1/2
//判断是否中文字符[\\u4e00-\\u9fa5]
def isChinese(one:Char):Boolean={return one >= 0x4e00 && one <= 0x9fa5;}
//将一段中英文本分为多行中文
def split(para:String):List[String] ={
    val ab=new ArrayBuffer[String]
    var start = -1;
    for(step <- 0 until para.length){
        //如果是汉字还未标记起点则标记
        if(isChinese(para(step))){
            if(start == -1){start = step}
        }
        else{
            if(start >= 0 ) {
                //找到一段汉字就输出，重置标志位
                ab += para.substring(start,step)
                start = -1
            }
        }
    }
    //最后一句
    if(start >= 0){ ab += para.substring(start,para.length)}    
    ab.toList
}
//清洗语料，原在一行的每片段汉字是单独成行，形成多行
def etl(path:String):RDD[String]={sc.textFile(path).flatMap(a => split(a)).filter(_.length > 1) } 
//清洗语料，原在一行的多片段汉字还按行保存，并标注BMES
def etl2(path:String):RDD[(String,String)]={
    sc.textFile(path).map(split(_)).map(_.map(b => (b,bmes(b)))).map(a => (a.map(_._1).mkString,a.map(_._2).mkString))
} 
def saveCorpus2(c:RDD[(String,String)],kind:String){
    val file1=s"d:/bigdata/ml/data/corpus1$kind.txt"
    val file2=s"d:/bigdata/ml/data/corpus2$kind.txt"
    for(one <- c){
        log(file1,one._1)
        log(file2,one._2)
    }
}
//转换语料
//saveCorpus2(etl2("d:/bigdata/ml/data/cnlc_train.txt"),"c")
saveCorpus2(etl2("d:/bigdata/ml/data/people2014_words.txt"),"p")

