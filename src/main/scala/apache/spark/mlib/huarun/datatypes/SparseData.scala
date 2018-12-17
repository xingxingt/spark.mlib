package apache.spark.mlib.huarun.datatypes

import org.apache.spark.{SparkConf, SparkContext}

/**
  * It is very common in practice to have sparse training data.
  * MLlib supports reading training examples stored in LIBSVM format,
  * which is the default format used by LIBSVM and LIBLINEAR.
  * It is a text format in which each line represents a labeled sparse feature vector using the following format:
  * label index1:value1 index2:value2 ...
  * where the indices are one-based and in ascending order.
  * After loading, the feature indices are converted to zero-based.
  */

object SparseData{
  def main(args: Array[String]): Unit = {
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.util.MLUtils
    import org.apache.spark.rdd.RDD
    val sparkConf = new SparkConf().setMaster("local").setAppName("SparseData")
    val sc = new SparkContext(sparkConf)
    val examples: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, "src/main/data/mllib/sample_libsvm_data.txt")

    examples.collect().toList.foreach(x=>println(x))
  }
}