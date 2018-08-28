//:load d:/bigdata/ml/tree.scala
case class BinTree[T](var value: T, var left: BinTree[T] = null, var right: BinTree[T] = null) {
    var depth=1
    var parent=this
    var isRight= true
    def _initAux(r:Boolean, d:Int,p:BinTree[T]){ isRight = r;depth = d;parent = p; }
    //增加左/右节点，注意是二叉树，重复增加为覆盖
    def addLeft(v: T) {left = BinTree(v);left._initAux(false,depth+1,this)}
    def addRight(v: T) {right = BinTree(v);right._initAux(true,depth+1,this)}
    //删除左/右节点
    def delLeft(){left = null}
    def delRight(){right = null}
    //是否为根结点
    def isRoot():Boolean = {parent == this}
    //是否为叶子结点
    def isLeaf():Boolean = {left == null && right == null}
    //有左/右节点
    def hasLeft():Boolean = {left != null}
    def hasRight():Boolean = {right != null}
    //获取第n个右子节点，n应该小于layer
    def getRight(n:Int):BinTree[T]= {var cur=this;for(a <- 1 to n){cur = cur.right};cur}
    //转化为List
    def flatten():List[T]= value :: {if ( hasRight ) right.flatten else Nil} ::: {if (hasLeft) left.flatten  else Nil }
    //树的节点数
    def size():Int = flatten.size
    //树的总深度
    def layer():Int= if(hasRight ) right.layer else if(hasLeft) left.layer else depth
    //深度克隆
    def _deepClone(root:BinTree[T]){
	    if(hasRight) {root.addRight(right.value);right._deepClone(root.right)}
	    if(hasLeft) {root.addLeft(left.value);left._deepClone(root.left)}
    }
    def deepClone():BinTree[T]={val root=BinTree(value);_deepClone(root);root}
    //通用遍历方法
    def _visit(f:(T,Int)=>Unit){
        f(value,depth)
        if(hasRight ) right._visit(f)
        if(hasLeft ) left._visit(f)
    } 
    //调试输出
    def show(){_visit({(v:T,h:Int) => (0 to h).foreach(a => print(" "));println(v)})}
}