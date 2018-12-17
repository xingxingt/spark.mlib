package apache.spark.mlib.huarun.datatypes

import org.apache.spark.rdd.RDD

object RowMatrix {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.mllib.linalg.Vector
    import org.apache.spark.mllib.linalg.distributed.RowMatrix

    val rows: RDD[Vector] =null // an RDD of local vectors
    // Create a RowMatrix from an RDD[Vector].
    val mat: RowMatrix = new RowMatrix(rows)

    // Get its size.
    val m = mat.numRows()
    val n = mat.numCols()

    // QR decomposition
    /**
      * QR decomposition is of the form A = QR
      * where Q is an orthogonal matrix and R is an upper triangular matrix.
      * For singular value decomposition (SVD) and principal component analysis (PCA),
      * please refer to Dimensionality reduction.
      */
    val qrResult = mat.tallSkinnyQR(true)

  }

}
