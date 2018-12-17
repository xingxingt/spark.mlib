package apache.spark.mlib.huarun.featureengineering
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.SparkSession

object CountVectorizerDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local[2]").getOrCreate()
    spark.sparkContext.setLogLevel("warn")
    val dataFrame = spark.createDataFrame(Seq(
      (0, Array("a", "b","b","c","d","d")),
      (1, Array("a","c","b" ))
    )).toDF("id", "words")

    //setVocabSize设定词汇表的最大容量为3，setMinDF设定词汇表中的词至少要在2个文档中出现过。
    //如果setMinDF=2 那么就不会出现d(只在一个文档存在)了。
    val cv = new CountVectorizer().setVocabSize(3).setMinDF(2).setInputCol("words").setOutputCol("features")
    //如果setVocabSize=2 那么就不会出现a,c(次数少)了。
    val cv1 = new CountVectorizer().setVocabSize(2).setInputCol("words").setOutputCol("features")

    val cvModel = cv.fit(dataFrame)
    val cvModel1 = cv1.fit(dataFrame)

    cvModel.transform(dataFrame).show(truncate = false)
    cvModel1.transform(dataFrame).show(truncate = false)

  }
}