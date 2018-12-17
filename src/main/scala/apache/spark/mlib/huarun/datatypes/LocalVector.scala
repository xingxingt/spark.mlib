package apache.spark.mlib.huarun.datatypes

import org.apache.spark.mllib.linalg.{Vector, Vectors}
/**
  * spark的底层的向量和矩阵是基于Breeze的,
  * 对于稠密向量：很直观，你要创建什么，就加入什么，其函数声明为Vector.dense(values : Array[Double])
  * 对于稀疏向量，当采用第一种方式时，3表示此向量的长度，第一个Array(0,2)表示的索引，
  * 第二个Array(1.0, 3.0)与前面的Array(0,2)是相互对应的，表示第0个位置的值为1.0，第2个位置的值为3
  *  对于稀疏向量，当采用第二种方式时，3表示此向量的长度，后面的比较直观，Seq里面每一对都是(索引，值）的形式
  */
object LocalVector {
  def main(args: Array[String]): Unit = {

    // Create a dense vector (1.0, 0.0, 3.0).
    val dv: Vector = Vectors.dense(1.0, 0.0, 3.0)
    println(dv)
    // Create a sparse vector (1.0, 0.0, 3.0) by specifying its indices and values corresponding to nonzero entries.
    val sv1: Vector = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
    // Create a sparse vector (1.0, 0.0, 3.0) by specifying its nonzero entries.
    //3 is the size of the vector.
    val sv2: Vector = Vectors.sparse(3, Seq((0, 1.0), (2, 3.0)))
    print(sv2)


  }






}
