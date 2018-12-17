package apache.spark.mlib.huarun.datatypes

/**
  * A labeled point is a local vector, either dense or sparse, associated with a label/response.
  * In MLlib, labeled points are used in supervised learning algorithms. We use a double to store a label,
  * so we can use labeled points in both regression and classification.
  * For binary classification, a label should be either 0 (negative) or 1 (positive).
  * For multiclass classification, labels should be class indices starting from zero: 0, 1, 2, ....

  */
object Labeledpoint{
  def main(args: Array[String]): Unit = {
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.regression.LabeledPoint
    // Create a labeled point with a positive label and a dense feature vector.
    val pos = LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0))

    // Create a labeled point with a negative label and a sparse feature vector.
    val neg = LabeledPoint(0.0, Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0)))

  }
}
