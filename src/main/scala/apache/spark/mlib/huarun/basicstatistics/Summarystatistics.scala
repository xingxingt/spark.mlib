package apache.spark.mlib.huarun.basicstatistics

import org.apache.spark.{SparkConf, SparkContext}

/**
  *
  * colStats() returns an instance of MultivariateStatisticalSummary,which contains the column-wise max, min, mean, variance, and number of nonzeros,
  * as well as the total count.
  * more learn:http://www.cnblogs.com/arachis/p/Similarity.html
  */
object SummaryStatistics {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
    val sparkConf = new SparkConf().setMaster("local").setAppName("Summarystatistics")
    val sc = new SparkContext(sparkConf)

    val observations = sc.parallelize(
      Seq(
        Vectors.dense(1.0, 10.0, 100.0),
        Vectors.dense(2.0, 20.0, 200.0),
        Vectors.dense(3.0, 30.0, 300.0)
      )
    )

    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean)  // a dense vector containing the mean value for each column
    println(summary.variance)  // column-wise variance
    println(summary.numNonzeros)  // number of nonzeros in each column

  }

}
