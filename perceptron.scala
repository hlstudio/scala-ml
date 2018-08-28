//:load d:/bigdata/ml/perceptron.scala
    val sourceX = Array(Array(3,3),Array(4,3),Array(1,1))
    val resulty = Array(1,1,-1)

    def perceptronV1(X:Array[Array[Int]],y:Array[Int],learnRate:Double = 1,iterNum:Int = 1000):(Array[Double],Double)={
            var theta:Array[Double] = new Array(X(0).length)
            var b = 0.0
            var quit = 1
            println("iter\t row\t lost\t w1\t w2\t b")
            for(i<- 0 to iterNum if quit > 0){
                    val row = i % X.length
                    var lost = 0.0
                    //模拟向量计算，X是个二维向量
                    for(col <- 0 until X(0).length){
                            lost += X(row)(col) * theta(col)
                    }
                    lost = (lost + b) * y(row)
                    if(!(lost > 0)){
                            for(col <- 0 until X(0).length){
                                    theta(col) += learnRate * y(row) * X(row)(col)
                            }
                            b = b + learnRate * y(row) 
                    }
                    println(s"$i\t $row\t $lost\t ${theta(0)}\t ${theta(1)}\t $b")
                    //收敛后就退出，加了个不准确的条件，应该是所有分类都正确
                    if(lost > 0 && learnRate * y(row) < 0.001) quit = -1 
            }
            (theta,b)
    }  
    perceptronV1(sourceX,resulty)

    def perceptronV2(X:Array[Array[Int]],y:Array[Int],learnRate:Double = 1):(Array[Double],Double)={
            var Array(w1,w2) = Array(0.0,0.0)
            var b = 0.0
            var quit = -1  //全部分类正确
            println("row\t lost\t w1\t w2\t b")
            while(quit < 0){
                    quit = 1
                    for(row <- 0 until X.length){
                       val Array(x1,x2)=X(row)
                        val y1=y(row)
                        val lost = (w1*x1+w2*x2 +b)*y1
                        if(lost <= 0){
                                w1 += learnRate*y1*x1
                                w2 += learnRate*y1*x2
                                b = b + learnRate * y1
                                quit = -1 
                        }
                        println(s"$row\t $lost\t $w1\t $w2\t $b")
                   }
            }
            (Array(w1,w2),b)
    }  
    perceptronV2(sourceX,resulty)

    def dot[T](a:Array[T],b:Array[T])(implicit m:Numeric[T]):T = {import m._;a.zip(b).map(t => t._1*t._2).sum}
    def add[T:Manifest](a:Array[T],b:Array[T])(implicit m:Numeric[T]):Array[T] = {import m._;a.zip(b).map(t => t._1+t._2)}
    def perceptronV3(X:Array[Array[Int]],y:Array[Int],learnRate:Double = 1):(Array[Double],Double)={
            var W = Array(0.0,0.0)
            var b = 0.0
            var quit = -1  //全部分类正确
            println("row\t lost\t w1\t w2\t b")
            while(quit < 0){
                    quit = 1
                    for(row <- 0 until X.length){
                        val Array(x1,x2)=X(row)
                        val y1=y(row)
                        val lost = (dot(W,X(row).map(_.toDouble)) +b)*y1
                        if(lost <= 0){
                                W = add(W,Array(learnRate*y1*x1,learnRate*y1*x2))
                                b = b + learnRate * y1
                                quit = -1 
                        }
                        println(s"$row\t $lost\t ${W(0)}\t ${W(1)}\t $b")
                   }
            }
            (W,b)
    }  
    perceptronV3(sourceX,resulty)

   //对偶算法
   def perceptronDualV1(X:Array[Array[Int]],y:Array[Int],learnRate:Double = 1):(Array[Double],Double)={
            var A = Array(0.0,0.0,0.0)
            var b = 0.0
            var quit = -1
            //计算gram矩阵
            val G=Array(Array(dot(X(0),X(0)),dot(X(0),X(1)),dot(X(0),X(2))),
                     Array(dot(X(1),X(0)),dot(X(1),X(1)),dot(X(1),X(2))),
                     Array(dot(X(2),X(0)),dot(X(2),X(1)),dot(X(2),X(2))))
            println("row\t lost\t a1\t a2\t a3\t b")
            while(quit < 0){
                    quit = 1
                    for(row <- 0 until X.length){
                       //对偶损失函数的三步推导 仿python (np.dot(a * y, Gram[i]) + b)*y[i]
                        //val lost = ((A(0)*X(0)*y(0)+A(1)*X(1)*y(1)+A(2)*X(2)*y(2))*X(row)+b)*y(row)
                        //         = ( A(0)*y(0)*X(0)*X(row) + A(1)*y(1)*X(1)*X(row) +A(2)*y(2)*X(2)*X(row) +b )*y(row)
                        val lost = ( A(0)*y(0)*G(0)(row)+A(1)*y(1)*G(1)(row)+A(2)*y(2)*G(2)(row) + b )*y(row)
                        if(lost <= 0){
                                A(row) += 1
                                b = b + learnRate * y(row)
                                quit = -1 
                        }
                        println(s"$row\t $lost\t ${A(0)}\t ${A(1)}\t ${A(2)}\t $b")
                   }
            }
            var W = Array(0.0,0.0)
            //w就是aixiyi的累加和 仿python w = np.dot(a * y, x)
            W(0)=A(0)*X(0)(0)*y(0)+A(1)*X(1)(0)*y(1)+A(2)*X(2)(0)*y(2)
            W(1)=A(0)*X(0)(1)*y(0)+A(1)*X(1)(1)*y(1)+A(2)*X(2)(1)*y(2)
            (W,b)
    }
    perceptronDualV1(sourceX,resulty) 

    import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}  
    def perceptronV4(XA:Array[Array[Int]],y:Array[Int],learnRate:Double = 1):(Array[Double],Double)={
            val X=BDM(XA(0).map(_.toDouble),XA(1).map(_.toDouble),XA(2).map(_.toDouble))
            var W = BDV.zeros[Double](X.cols)
            var b = 0.0
            var quit = -1  //全部分类正确
            println("row\t lost\t w1\t w2\t b")
            while(quit < 0){
                    quit = 1
                    for(row <- 0 until X.rows){
                        val xrow=X(row,::).t
                        val lost = (xrow.dot(W) +b)*y(row)
                        if(lost <= 0){
                                W += xrow * (learnRate*y(row))
                                b += learnRate * y(row)
                                quit = -1 
                        }
                        println(s"$row\t $lost\t ${W(0)}\t ${W(1)}\t $b")
                   }
            }
            (W.toArray,b)
    }  
    perceptronV4(sourceX,resulty)  

   def perceptronDualV2(XA:Array[Array[Int]],y:Array[Int],learnRate:Double = 1):(Array[Double],Double)={
            val X=BDM.zeros[Double](XA.length,XA(0).length)
            for(i <- 0 until X.rows){for (j <- 0 until X.cols){ X(i,j)=XA(i)(j).toDouble}}
            val Y=BDV(y.map(_.toDouble))
            var W = BDV.zeros[Double](X.cols)
            var A = BDV.zeros[Double](X.rows)
            var b = 0.0
            var quit = -1
            //计算gram矩阵
            val G=BDM.zeros[Double](X.rows,X.rows)
            for(i <- 0 until X.rows){for (j <- 0 until X.rows) { G(i,j)= X(i,::).dot(X(j,::))}}
            println("row\t lost\t a1\t a2\t a3\t b")
            while(quit < 0){
                    quit = 1
                    for(row <- 0 until X.rows){
                        val lost = (G(row,::).t.dot( A :* Y) + b ) * y(row)
                        if(lost <= 0){
                                A(row) += 1
                                b += learnRate * y(row)
                                quit = -1 
                        }
                        println(s"$row\t $lost\t ${A(0)}\t ${A(1)}\t ${A(2)}\t $b")
                   }
            }
            for(i <- 0 until X.cols){W(i) = X(::,i).dot( A :* Y)}
            (W.toArray,b)
    }
    perceptronDualV2(sourceX,resulty)
/*
perceptronDualV2: (XA: Array[Array[Int]], y: Array[Int], learnRate: Double)(Array[Double], Double)
row      lost    a1      a2      a3      b
0        0.0     1.0     0.0     0.0     1.0
1        22.0    1.0     0.0     0.0     1.0
2        -7.0    1.0     0.0     1.0     0.0
0        12.0    1.0     0.0     1.0     0.0
1        14.0    1.0     0.0     1.0     0.0
2        -4.0    1.0     0.0     2.0     -1.0
0        5.0     1.0     0.0     2.0     -1.0
1        6.0     1.0     0.0     2.0     -1.0
2        -1.0    1.0     0.0     3.0     -2.0
0        -2.0    2.0     0.0     3.0     -1.0
1        20.0    2.0     0.0     3.0     -1.0
2        -5.0    2.0     0.0     4.0     -2.0
0        10.0    2.0     0.0     4.0     -2.0
1        12.0    2.0     0.0     4.0     -2.0
2        -2.0    2.0     0.0     5.0     -3.0
0        3.0     2.0     0.0     5.0     -3.0
1        4.0     2.0     0.0     5.0     -3.0
2        1.0     2.0     0.0     5.0     -3.0
res5: (Array[Double], Double) = (DenseVector(1.0, 1.0),-3.0)
*/  
