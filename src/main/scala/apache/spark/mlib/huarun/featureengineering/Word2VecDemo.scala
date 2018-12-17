package apache.spark.mlib.huarun.featureengineering

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

object Word2VecDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[2]").appName("Word2VecDemo").getOrCreate()
    val documentDF = spark.createDataFrame(Seq(
      "Hi I love Spark".split(" "),
      "Hi I love java".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")
    // setVectorSize 目标数值向量的维度大小 setMinCount 只有当某个词出现的次数大于或者等于 minCount 时，才会被包含到词汇表里，否则会被忽略掉
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)
    //​利用Word2VecModel把文档转变成特征向量。
    val result = model.transform(documentDF)
    result.show(false)
  }
}
