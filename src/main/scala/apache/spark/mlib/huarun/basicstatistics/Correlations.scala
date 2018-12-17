package apache.spark.mlib.huarun.basicstatistics

import org.apache.spark.{SparkConf, SparkContext}

object Correlations {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.mllib.stat.Statistics
    import org.apache.spark.rdd.RDD
    val sparkConf = new SparkConf().setMaster("local").setAppName("Correlations")
    val sc = new SparkContext(sparkConf)

    val seriesX: RDD[Double] = sc.parallelize(Array(1, 2, 3, 3, 5))  // a series
    // must have the same number of partitions and cardinality as seriesX
    val seriesY: RDD[Double] = sc.parallelize(Array(11, 22, 33, 33, 555))

    // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method. If a
    // method is not specified, Pearson's method will be used by default.
    val correlation: Double = Statistics.corr(seriesX, seriesY, "pearson")
    println(s"Correlation is: $correlation")

    val correlation1: Double = Statistics.corr(seriesX, seriesY, "spearman")
    println(s"Correlation1 is: $correlation1")

    val data: RDD[Vector] = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(5.0, 33.0, 366.0))
    )  // note that each Vector is a row and not a column

    // calculate the correlation matrix using Pearson's method. Use "spearman（斯皮尔曼）" for Spearman's method
    // If a method is not specified, Pearson's method will be used by default.

    val correlMatrix: Matrix = Statistics.corr(data, "pearson")

    println(correlMatrix.toString)
  }

}
