//:load d:/bigdata/ml/crf.scala
import scala.collection.mutable.ArrayBuffer
import java.io._
import org.apache.spark.rdd._
import scala.reflect.ClassTag

//��hmm��ģ�ͻ���֮�ϣ�����2���۲�����ת�ƾ���obs_pre_p��obs_next_p����ʵ��ֻ��ǰһ�ַ���һ�ַ����
//����hmm model states ��״̬  start_p ��ʼ���ʣ���״̬�� trans_p ת�Ƹ��ʣ���״̬�� emit_p ������� ����״̬����Ϊ��״̬�ĸ��ʣ�
/*
val (states,chars,start_p,trans_p,emit_p,obs_pre_p,obs_next_p)=(crf.states,crf.chars,crf.start_p,crf.trans_p,crf.emit_p,crf.obs_pre_p,crf.obs_next_p)
val obs=test(0).map(chars.indexOf(_))
*/
case class CRFModel(var states:Array[Int],var chars:Array[Char],var start_p:Array[Double],var trans_p:Array[Array[Double]],var emit_p:Array[Array[Double]],var obs_pre_p:Array[Array[Array[Double]]],var obs_next_p:Array[Array[Array[Double]]]){
    //���crfģ�� ����@see https://github.com/hankcs/Viterbi/blob/master/src/com/hankcs/algorithm/Viterbi.java 
    def viterbi(obs:Array[Int]):Array[Int]={
        val V = Array.ofDim[Double](obs.length,states.length)
        var path = Array.ofDim[Int](states.length,obs.length)
        for (y <- states){
            V(0)(y) = start_p(y) *  (emit_p(y)(obs(0)) + obs_next_p(y)(obs(0))(obs(1))) 
            path(y)(0) = y
        }
        
        for (t <- 1 until obs.length ){
            val newpath = Array.ofDim[Int](states.length,obs.length)
            for (y <- states){
                val (state,prob)={for(y0 <- states) yield (y0,V(t - 1)(y0) * trans_p(y0)(y)  * ( emit_p(y)(obs(t)) +  obs_pre_p(y)(obs(t))(obs(t-1)) + {if(t < obs.length -1){obs_next_p(y)(obs(t))(obs(t+1))}else{0}}) )}.maxBy(_._2)
                V(t)(y) = prob 
                System.arraycopy(path(state), 0, newpath(y), 0, t)
                newpath(y)(t) = y
            }
            path = newpath
        }
        val (state,prob)={for(y <- states) yield(y,V(obs.length - 1)(y))}.maxBy(_._2)
        path(state)
    }
    //����BMES���ɷִʽ��
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
    
    //����crfģ�ͷִ�
    def seg(str:String):List[String]={
        val bmes=Array('B','M','E','S')
        val result=viterbi(str.map(chars.indexOf(_)).toArray).map(bmes(_))
        eject(str,result)
    }

    //����ģ�ͣ�TODO:��Ӧ��д����������Ӧ��ֻд�봿���ݲ��֣������޸ķ������������л��İ汾�������load����
    def save(path:String){
        val oos = new ObjectOutputStream(new FileOutputStream(path))
        oos.writeObject(this)
        oos.close()        
    }
}

//����ģ��
def load(path:String):CRFModel={
    val ois = new ObjectInputStream(new FileInputStream(path))
    val h=ois.readObject.asInstanceOf[CRFModel]
    ois.close
    h
}

//N-GRAMԪ�ִ�
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

//��ȡ�Ĺ��÷��������¸�������
def updateArray(dic:Array[(Char, Int)],chars:Array[Char],rowIndex:Int,array:Array[Array[Double]]){
    val rowTotal=dic.map(_._2).sum
    dic.foreach(column =>{
        val columnIndex=chars.indexOf(column._1)
        val ratio=1.0d*column._2 / rowTotal
        array(rowIndex)(columnIndex) = ratio
    })    
}

//���ݱ�ע��������crfģ��,file1Ϊ���ģ�file2ΪBMES��ע
def train(file1:String,file2:String):CRFModel={
    val c1=sc.textFile(file1)
    val c2=sc.textFile(file2)
    //�ܳ���
    val total=c1.map(_.size).sum
    //�۲⼯����
    val chars=c1.flatMap(_.toString).distinct.collect.sorted
    //����г�ʼ����
    val pi=c2.flatMap(_.toString).map((_,1)).reduceByKey(_+_).map(a =>(a._1,1.0d*a._2/total)).collect
    //˳��ΪBMES
    val bmes=Array('B','M','E','S')
    val states=Array(0,1,2,3)
    //��ʼ����pi
    val start_p=pi.sorted.map(_._2)
    val temp=start_p(2);start_p(2)=start_p(1);start_p(1)=temp

    //bmes��ת�Ƹ���A
    val trans=c2.flatMap(a => ngram(a,2)).reduceByKey(_+_)
    val trans_p=Array.ofDim[Double](states.length,states.length)
    for(row <- bmes){
        val rowIndex=bmes.indexOf(row)
        val dic=trans.filter(_._1(0) == row).map(a =>(a._1(1),a._2)).reduceByKey(_+_).collect
        updateArray(dic,bmes,rowIndex,trans_p)
    }

    //�������B,�ȴ�����zipShuffle��zip�������ܼ�ʮ��
    val c1c2=c1.zipShuffle(c2)
    val emit=c1c2.flatMap(a => a._1.zip(a._2))
    val emit_p=Array.ofDim[Double](states.length,chars.length)
    for(row <- bmes){
        val rowIndex=bmes.indexOf(row)
        val dic=emit.filter(_._2 == row).map(a =>(a._1,1)).reduceByKey(_+_).collect
        updateArray(dic,chars,rowIndex,emit_p)
    }
    
    //�۲����֮���ת�Ƹ��ʣ�crf��hmm�����������
    val obs=c1c2.flatMap(a => a._1.sliding(2).zip(a._2.sliding(2))).filter(a =>a._1.length  == 2 && a._2.length == 2)
    //TODO:2�����3ά����һ�������OOM��,�ɿ���sparsematrix
    val obs_pre_p=Array.ofDim[Double](states.length,chars.length,chars.length)
    val obs_next_p=Array.ofDim[Double](states.length,chars.length,chars.length)
    for(page <- bmes){
        val pageIndex=bmes.indexOf(page)
        val dicNext=obs.filter(_._2(0) == page).map(a =>(a._1,1)).reduceByKey(_+_).groupBy(_._1(0))
        dicNext.collect.foreach(row =>{
            val rowIndex=chars.indexOf(row._1)
            val dic=row._2.map(a =>(a._1(1),a._2)).groupBy(_._1).map(a =>(a._1,a._2.map(_._2).sum)).toArray
            updateArray(dic,chars,rowIndex,obs_next_p(pageIndex))
        })
        val dicPre=obs.filter(_._2(1) == page).map(a =>(a._1,1)).reduceByKey(_+_).groupBy(_._1(1))
        dicPre.collect.foreach(row =>{
            val rowIndex=chars.indexOf(row._1)
            val dic=row._2.map(a =>(a._1(0),a._2)).groupBy(_._1).map(a =>(a._1,a._2.map(_._2).sum)).toArray
            updateArray(dic,chars,rowIndex,obs_pre_p(pageIndex))
        })
    }
    //���ˣ�crf�Ĳ����������
    CRFModel(states, chars, start_p, trans_p,emit_p,obs_pre_p,obs_next_p)
}

val (file1,file2)=("d:/bigdata/ml/data/corpus1p.txt","d:/bigdata/ml/data/corpus2p.txt")
val crf=train(file1,file2)
val test=Array("�Ļ������η�չ���������ο͵����������������о���",
"��ģ�Ͳ�������ѵ���ڷִʽ׶���ͨ��ģ�ͼ�����ִַʳ��ֵĸ���",
"��ѧƽ�ȱ�ʾ�ᱣ֤�������İ�ȫ",
"����һ���ֻ�����",
"�ͼֱ�һ����λ�Ķ������Ѿ�������������������ڼֱ��Ұ�������׼����ٷ���",
"��ҹ���º���������һ׮����Ů�������ڳ�����ʧ����",
"�������н�������Զ���",
"��Ī����˹�����׶صĽ�˹�ع�˾�ϰ�",
"������̥�����ƫ�����ƫӲ������������",
"������ǿ��ͥ�����ƻ����׸������û�и����������ǲ�������")
if(file1.endsWith("corpus1.txt")){crf.seg(test(0))}else{test.map(crf.seg(_)).foreach(println)}
