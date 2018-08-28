//:load d:/bigdata/ml/corpus.scala
import org.apache.spark.rdd.RDD
import scala.util.Random
import scala.collection.mutable.ArrayBuffer

val homeDir= "c:/temp/raw/data"
//����ʵ�
def transDict():RDD[(String,Int)]={
    val han=sc.textFile(s"${homeDir}/hanlp/*.txt")
    //ѡ��ʮԪ�ʼ�����
    val one=han.map(_.split("[ \t]")).filter(a => !a(0).contains("##") && a(0).length < 11 && a.length < 8).map(a =>a match {
        case Array(x1,y1,z1) => (x1,y1,z1.toInt)
        case Array(x1,y1,z1,y2,z2) => (x1,y1,z1.toInt+z2.toInt)
        case Array(x1,y1,z1,y2,z2,y3,z3) => (x1,y1,z1.toInt+z2.toInt+z3.toInt)
    })
    //�������ԣ�ȥ��
    one.map(a =>(a._1,a._3)).distinct
}

//�����б�ע
def bmes(key:String):String={
    key.length match {
        case 1 => "S"
        case 2 => "BE"
        case _ => "B"+List().padTo(key.length -2,"M").mkString+"E"
    }
}

//Ȩ������
def weightArray():Array[Int]={
    val a=new Array[Int](1000)
    a(0)=0
    for( i <- 1 until 1000) a(i)=a(i-1)+i
    a
}

//���ֲ���
def binarySearch[A <% Ordered[A]](a: IndexedSeq[A], v: A) :Int = {
  def recurse(low: Int, high: Int): Int = (low + high) / 2 match {
    case _ if high < low => high
    case mid if a(mid) > v => recurse(low, mid - 1)
    case mid if a(mid) < v => recurse(mid + 1, high)
    case mid => mid
  }
  recurse(0, a.size - 1)
}

//���ȡ��
def randWord(cand:Map[Int,List[String]],index:Int):String={
    //ȡ��ӦȨ����
    val group=cand(index+1)
    //Ȼ�����ȡһ����
    group(Random.nextInt(group.size))
}

//Ȩ�س�ʣ�lenΪ�������
def makeCorpus(len:Int):(RDD[(String,String)])={
    val wa=weightArray
    val one=transDict
    val cand=one.map(a => (a._1,if (a._2 > 1000) 1000 else a._2)).groupBy(_._2).map(a=>(a._1,a._2.toList.map(_._1))).collectAsMap.toMap
    val mm=(1 to 1000).sum
    //�����ȡ
    val r1=sc.makeRDD(1 to len).map(a => Random.nextInt(mm)).map(a => binarySearch(wa,a)).map(a => randWord(cand,a))
    //ԭʼ��ȫ���������
    val r2=sc.makeRDD(one.map(_._1).takeSample(false,one.count.toInt))
    val r=r1 ++ r2
    r.map(a =>(a,bmes(a)))
}
def log(filename:String,s:String){scala.tools.nsc.io.File(filename).appendAll(s+"\n")}
//��������
def saveCorpus(c:RDD[(String,String)]){
    //ƽ������Ϊ3.2����ÿ8��������һ������������
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

//���ݴ���Ƶ�ʿ���������
//saveCorpus(makeCorpus(80000*10000))

//ת�����ϣ������� �״�/n ��/v ����/v ��/r ��/w  ת��Ϊcorpus1/2
//�ж��Ƿ������ַ�[\\u4e00-\\u9fa5]
def isChinese(one:Char):Boolean={return one >= 0x4e00 && one <= 0x9fa5;}
//��һ����Ӣ�ı���Ϊ��������
def split(para:String):List[String] ={
    val ab=new ArrayBuffer[String]
    var start = -1;
    for(step <- 0 until para.length){
        //����Ǻ��ֻ�δ����������
        if(isChinese(para(step))){
            if(start == -1){start = step}
        }
        else{
            if(start >= 0 ) {
                //�ҵ�һ�κ��־���������ñ�־λ
                ab += para.substring(start,step)
                start = -1
            }
        }
    }
    //���һ��
    if(start >= 0){ ab += para.substring(start,para.length)}    
    ab.toList
}
//��ϴ���ϣ�ԭ��һ�е�ÿƬ�κ����ǵ������У��γɶ���
def etl(path:String):RDD[String]={sc.textFile(path).flatMap(a => split(a)).filter(_.length > 1) } 
//��ϴ���ϣ�ԭ��һ�еĶ�Ƭ�κ��ֻ����б��棬����עBMES
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
//ת������
//saveCorpus2(etl2("d:/bigdata/ml/data/cnlc_train.txt"),"c")
saveCorpus2(etl2("d:/bigdata/ml/data/people2014_words.txt"),"p")

