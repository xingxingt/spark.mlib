package apache.spark.mlib.huarun.basicstatistics

import org.apache.spark.{SparkConf, SparkContext}

object StratifiedSampling {
  def main(args: Array[String]): Unit = {


    val sparkConf = new SparkConf().setMaster("local").setAppName("StratifiedSampling")
    val sc = new SparkContext(sparkConf)
    // an RDD[(K, V)] of any key value pairs
    val data = sc.parallelize(
      Seq((1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (2, 'e'), (3, 'f')))

    // specify the exact fraction desired from each key
    val fractions = Map(1 -> 0.1, 2 -> 0.6, 3 -> 0.3)

    // Get an approximate sample from each stratum
    val approxSample = data.sampleByKey(withReplacement = false, fractions = fractions)
    approxSample.collect().toList.map(x=>println(x))
    // Get an exact sample from each stratum
    val exactSample = data.sampleByKeyExact(withReplacement = false, fractions = fractions)
    exactSample.collect().toList.map(x=>println(x))

  }
}
