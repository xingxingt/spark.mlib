package apache.spark.mlib.huarun.featureengineering

import org.apache.spark.ml.feature.Binarizer
import org.apache.spark.sql.SparkSession
/**
  * Created by liuyanling on 2018/3/19
  */
object BinarizerDemo {
  def main(args: Array[String]): Unit = {
    var spark = SparkSession.builder().appName("BinarizerDemo").master("local[2]").getOrCreate();
    val array = Array((1,34.0),(2,56.0),(3,58.0),(4,23.0))
    //将数组转为DataFrame
    val df = spark.createDataFrame(array).toDF("id","age")
    //初始化Binarizer对象并进行设定：setThreshold是设置我们的阈值，InputCol是设置需要进行二值化的输入列，setOutputCol设置输出列
    val binarizer = new Binarizer().setThreshold(50.0).setInputCol("age").setOutputCol("binarized_feature")
    //transform方法将DataFrame二值化。
    val binarizerdf = binarizer.transform(df)
    //show是用于展示结果
    binarizerdf.show
  }
}
