package org.apache.spark.examples.sql

import org.apache.spark.sql.SparkSession

object broadcastTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("Broadcast Test").master("local")
      .getOrCreate()
    val sc=spark.sparkContext
    val broadcastValues = sc.broadcast("test")

    val rdd1=sc.parallelize(1 to 100,10).mapPartitions{ p=>
      {
       val v= broadcastValues.value

        p.map(x=>x+v)
      }
    }
    rdd1.collect().toList.foreach(println)
    spark.close()


  }

}
