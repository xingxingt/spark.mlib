package apache.spark.mlib.huarun.featureengineering
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
/**
  * Created by LYL on 2018/4/4.
  */
object TFDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("TF-IDF Demo").master("local").getOrCreate()
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "china kungfu kungfu is good"),
      (1.0, "I lova china"),
      (2.0, "I love china shenzhen")
    )).toDF("label", "sentence")
    //Tokenizer分词器 将句子分成单词
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    //将每个词转换成Int型，并计算其在文档中的词频（TF）
    //setNumFeatures(200)表示将Hash分桶的数量设置为200个,可以根据你的词语数量来调整,一般来说，
    //这个值越大不同的词被计算为一个Hash值的概率就越小，数据也更准确，但需要消耗更大的内存
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("TF Features").setNumFeatures(200)
    val featurizedData = hashingTF.transform(wordsData)
    //计算IDF
    val idf = new IDF().setInputCol("TF Features").setOutputCol("TF-IDF features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("words","TF Features","TF-IDF features").show(false)
  }
}