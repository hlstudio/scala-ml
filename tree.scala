//:load d:/bigdata/ml/tree.scala
case class BinTree[T](var value: T, var left: BinTree[T] = null, var right: BinTree[T] = null) {
    var depth=1
    var parent=this
    var isRight= true
    def _initAux(r:Boolean, d:Int,p:BinTree[T]){ isRight = r;depth = d;parent = p; }
    //������/�ҽڵ㣬ע���Ƕ��������ظ�����Ϊ����
    def addLeft(v: T) {left = BinTree(v);left._initAux(false,depth+1,this)}
    def addRight(v: T) {right = BinTree(v);right._initAux(true,depth+1,this)}
    //ɾ����/�ҽڵ�
    def delLeft(){left = null}
    def delRight(){right = null}
    //�Ƿ�Ϊ�����
    def isRoot():Boolean = {parent == this}
    //�Ƿ�ΪҶ�ӽ��
    def isLeaf():Boolean = {left == null && right == null}
    //����/�ҽڵ�
    def hasLeft():Boolean = {left != null}
    def hasRight():Boolean = {right != null}
    //��ȡ��n�����ӽڵ㣬nӦ��С��layer
    def getRight(n:Int):BinTree[T]= {var cur=this;for(a <- 1 to n){cur = cur.right};cur}
    //ת��ΪList
    def flatten():List[T]= value :: {if ( hasRight ) right.flatten else Nil} ::: {if (hasLeft) left.flatten  else Nil }
    //���Ľڵ���
    def size():Int = flatten.size
    //���������
    def layer():Int= if(hasRight ) right.layer else if(hasLeft) left.layer else depth
    //��ȿ�¡
    def _deepClone(root:BinTree[T]){
	    if(hasRight) {root.addRight(right.value);right._deepClone(root.right)}
	    if(hasLeft) {root.addLeft(left.value);left._deepClone(root.left)}
    }
    def deepClone():BinTree[T]={val root=BinTree(value);_deepClone(root);root}
    //ͨ�ñ�������
    def _visit(f:(T,Int)=>Unit){
        f(value,depth)
        if(hasRight ) right._visit(f)
        if(hasLeft ) left._visit(f)
    } 
    //�������
    def show(){_visit({(v:T,h:Int) => (0 to h).foreach(a => print(" "));println(v)})}
}