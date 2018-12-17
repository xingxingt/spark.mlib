package apache.spark.mlib.huarun.datatypes

/**
  * A CSCMatrix has 3 arrays: a column pointers(Compressed Sparse Column) array called colPtr,
  * a rowIndices array, and a data array.
  * colPtr is of length cols + 1 (the number of columns + 1).
  * colPtr(c) is the index into rowIndices and data where the elements for the c'th column are stored, and colPtr(c+1) is the index beyond the last element. colPtr.last is always equal to rowIndices.length. rowIndices and data are parallel arrays, containing the row index and value for that entry in the matrix (surprising, I know). Row indices between colPtr(c) and colPtr(c+1) are always sorted.
  */

object Localmatrix {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.mllib.linalg.{Matrix, Matrices}

    // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    val dm: Matrix = Matrices.dense(3, 2, Array(1.0, 3.0, 5.0, 2.0, 4.0, 6.0))

    // Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
    /**
      * （1）3,2表示行列数。
      * （2）Array(0,2,3)，0表示开始迭代，1表示第一列非0元素个数，3表示第二列与第一列的非0元素个数累加和。
      * （3）Array(0,2,1)表示非0元素分别对应的行号。
      * （4）Array(9,6,8))表示非0元素。
      */
    val sm: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))
    val sm1: Matrix = Matrices.sparse(3, 2, Array(0, 1, 3), Array(0, 2, 1), Array(9, 6, 8))

    println(sm)
  }

}
