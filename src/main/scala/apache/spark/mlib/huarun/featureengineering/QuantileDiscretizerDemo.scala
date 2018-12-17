package apache.spark.mlib.huarun.featureengineering

import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.SparkSession

/**
  * 自动分项
  */
object QuantileDiscretizerDemo {
  def main(args: Array[String]): Unit = {
    var spark = SparkSession.builder().appName("QuantileDiscretizerDemo").master("local[2]").getOrCreate();
    val array = Array((1,13.0),(2,14.0),(3,22.0),(4,35.0),(5,44.0),(6,56.0),(7,21.0))
    val df = spark.createDataFrame(array).toDF("id","age")
    //和Bucketizer类似：将连续数值特征转换离散化。但这里不再自己定义splits（分类标准），而是定义分几箱就可以了。
    val quantile = new QuantileDiscretizer().setNumBuckets(5).setInputCol("age").setOutputCol("quantile_feature")
    //因为事先不知道分桶依据，所以要先fit,相当于对数据进行扫描一遍，取出分位数来，再transform进行转化。
    val quantiledf = quantile.fit(df).transform(df)
    //show是用于展示结果
    quantiledf.show
  }
}