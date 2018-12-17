package apache.spark.mlib.huarun.featureengineering

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
/**
  * Created by LYL on 2018/3/20.
  */
object MinMaxScalerDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("MinMaxScalerDemo").master("local[2]").getOrCreate()
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(2.0, 0.9, -2.0)),
      (1, Vectors.dense(3.0, -2.0, -3.0)),
      (2, Vectors.dense(4.0, -2.0, 2.0))
    )).toDF("id", "features")
    val maxabs = new MinMaxScaler().setInputCol("features").setOutputCol("minmax_features")
    val scalerModel = maxabs.fit(dataFrame)
    // 将所有值都缩放到[0,1]范围内
    val scalerdf = scalerModel.transform(dataFrame)
    scalerdf.show
  }
}