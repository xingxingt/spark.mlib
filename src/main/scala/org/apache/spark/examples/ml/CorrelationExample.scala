/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
// $example off$
import org.apache.spark.sql.SparkSession

/**
  * An example for computing correlation matrix.
  * Run with
  * {{{
  * bin/run-example ml.CorrelationExample
  * }}}
  */
object CorrelationExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CorrelationExample")
      .master("local[2]")
      .getOrCreate()
    import spark.implicits._

    // $example on$
    //todo  构建数据
    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
      Vectors.dense(4.0, 5.0, 0.0, 3.0),
      Vectors.dense(6.0, 7.0, 0.0, 8.0),
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
    )

    val df = data.map(Tuple1.apply).toDF("features")
    println("features dataframe:")
    df.show()

    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
    println(s"Pearson correlation matrix:\n $coeff1")
    //todo coeff1结果:每一行都有四个数，代表当前第几个向量与Seq中的4个向量的相关性
    //todo 例如:    1.0(第一个向量与第一个向量的相关性为1，因为他比较的是自己)      0.055641488407465814（第一个向量与第二个向量的相关性）  NaN（表示第一个向量和第二个向量没有相关性）  0.4004714203168137（第一个向量和第四个向量的相关性）
    //1.0                   0.055641488407465814  NaN  0.4004714203168137
    //0.055641488407465814  1.0                   NaN  0.9135958615342522
    //NaN                   NaN                   1.0  NaN
    //0.4004714203168137    0.9135958615342522    NaN  1.0


    //todo  使用指定的方法计算输入数据集的相关矩阵 默认Pearson
    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println(s"Spearman correlation matrix:\n $coeff2")
    // $example off$

    spark.stop()
  }
}

// scalastyle:on println
