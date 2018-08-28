//:load d:/bigdata/ml/cart.scala
:reset
:load d:/bigdata/ml/data.scala
:load d:/bigdata/ml/tree.scala

import scala.collection.mutable.ArrayBuffer

//贷款申请样本数据表的数值化
val train=List((0.0,Array(1.0 ,2.0 ,2.0 ,3.0 )),(0.0,Array(1.0 ,2.0 ,2.0 ,2.0 )),(1.0,Array(1.0 ,1.0 ,2.0 ,2.0 )),(1.0,Array(1.0 ,1.0 ,1.0 ,3.0 )),(0.0,Array(1.0 ,2.0 ,2.0 ,3.0 )),(0.0,Array(2.0 ,2.0 ,2.0 ,3.0 )),(0.0,Array(2.0 ,2.0 ,2.0 ,2.0 )),(1.0,Array(2.0 ,1.0 ,1.0 ,2.0 )),(1.0,Array(2.0 ,2.0 ,1.0 ,1.0 )),(1.0,Array(2.0 ,2.0 ,1.0 ,1.0 )),(1.0,Array(3.0 ,2.0 ,1.0 ,1.0 )),(1.0,Array(3.0 ,2.0 ,1.0 ,2.0 )),(1.0,Array(3.0 ,1.0 ,2.0 ,2.0 )),(1.0,Array(3.0 ,1.0 ,2.0 ,1.0 )),(0.0,Array(3.0 ,2.0 ,2.0 ,3.0 )))
val test=train.take(5)

val (train,test)=mnist

//寻找数据集的最佳切分点，返回值是(axis,value)
def bestSplit(list:List[LabeledArray]):(Int,Double)={
    val gini=for(index <- 0 until list(0)._2.size) yield{
        //特征1的所有可能取值分布
        val xd=list.map(a =>(a._2.apply(index),1)).groupBy(_._1).mapValues(_.size.toDouble)
        //特征1的所有可能取值在各类的分布
        val xp=list.map(a =>(a._1,a._2.apply(index))).groupBy(_._1).mapValues(_.map(a=>(a._2,1)).groupBy(_._1).mapValues(_.size.toDouble))
        //特征1的所有取值及gini
        val xa=for(p <- xd.keys) yield{
            val pp= for(a <- xp)yield a._2.getOrElse(p,0.0d)
            val (d,d1,d2)=(list.size.toDouble,pp.sum,list.size.toDouble-pp.sum)
            val g1= 1 - pp.map(a => math.pow(a/pp.sum,2)).sum
            //去掉特征1的第1个取值后的类别分布
            val remainxp=list.map(a =>(a._1,a._2.apply(index))).filter(_._2 != p).groupBy(_._1).mapValues(_.size.toDouble).map(_._2)
            val g2= 1 - remainxp.map(a => math.pow(a / remainxp.sum,2)).sum
            //特征1的第1个取值的gini指数
            val gini=d1/d*g1 + d2/d*g2
            (p,gini)
        }
        (index,xa)
    }
    val best=gini.map(a =>(a._1,a._2.toArray.sortBy(_._2).head)).map(a =>(a._1,a._2._1,a._2._2)).sortBy(_._3).head
    (best._1,best._2)
}

//根据切分点切分数据集
def split(list:List[LabeledArray],line:(Int,Double)):(List[LabeledArray],List[LabeledArray])={
    val left=list.filter(_._2.apply(line._1) == line._2)
    val right=list.filter(_._2.apply(line._1) != line._2)
    (left,right)
}

//规则为(axis,value,label)
def createTree(list:List[LabeledArray],rule:BinTree[(Int,Double,Double)]){
    val line=bestSplit(list)
    val (left,right)=split(list,line)
    val leftResult=left.map(_._1).distinct
    val rightResult=right.map(_._1).distinct
    //获得划分，增加规则
    if(leftResult.size == 1) {rule.addLeft((line._1,line._2,leftResult(0))) }
    if(rightResult.size == 1){rule.addRight((line._1,line._2,rightResult(0))) } 
    //左子树处理
    if(leftResult.size > 1) {rule.addLeft((line._1,line._2,-1.0)) ; createTree(left,rule.left) }
    //右子树处理
    if(rightResult.size > 1) {rule.addRight((line._1,line._2,-1.0)) ; createTree(right,rule.right) }
}

//剪枝，未进行最优选择
def pruning(rule:BinTree[(Int,Double,Double)]):List[BinTree[(Int,Double,Double)]]={
    val all=for(layer <- rule.layer /2 until rule.layer -1) yield{
        val r=rule.deepClone
        val cur=r.getRight(layer)
        //更新类别
        cur.value=(cur.value._1,cur.value._2,cur.flatten.filter(_._3 != -1).groupBy(_._3).maxBy(_._2.size)._1)
        cur.delRight
        cur.delLeft
        r
    }
    rule :: all.toList
}

//预测
def _predicate(x:Array[Double],rule:BinTree[(Int,Double,Double)],result:ArrayBuffer[Double]){
    val nl=rule.left
    if( rule.hasLeft && x(nl.value._1) == nl.value._2){
        if(nl.isLeaf){result += nl.value._3} else _predicate(x,nl,result)
    }
    val nr=rule.right
    if( rule.hasRight && x(nr.value._1) != nr.value._2){
        if(nr.isLeaf){result += nr.value._3} else _predicate(x,nr,result)
    }
}

def predicate(x:Array[Double],rule:BinTree[(Int,Double,Double)]):Double={
    val label=new ArrayBuffer[Double]
    _predicate(x,rule,label)
    if(label.isEmpty) -1.0 else label(0)
}


val rule=BinTree((-1,-1.0,-1.0))
createTree(train,rule)
for(rule2 <- pruning(rule)){
    val acc1=1.0d * train.map(a => (a._1,predicate(a._2,rule2))).filter(a => a._1 == a._2).size /train.size
    val acc2=1.0d * test.map(a => (a._1,predicate(a._2,rule2))).filter(a => a._1 == a._2).size /test.size
    println(s"rule tree layer:${rule2.layer}, nodes:${rule2.size}, Accuracy test:$acc2, train:$acc1")
}