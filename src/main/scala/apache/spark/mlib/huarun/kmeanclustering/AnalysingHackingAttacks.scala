package apache.spark.mlib.huarun.kmeanclustering

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types._

/**
  * https://medium.com/tensorist/using-k-means-to-analyse-hacking-attacks-81957c492c93
  */

object AnalysingHackingAttacks {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("AnalysingHackingAttacks")
      .master("local[2]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("warn")

    import spark.implicits._

    //load data
    /**
      * Session_Connection_Time (How long the session lasted in minutes)
      * Bytes_Transferred (Megabytes transferred during session)
      * Kali_Trace_Used (Whether the hacker was using Kali Linux)
      * Servers_Corrupted (Number of server corrupted during the attack)
      * Pages_Corrupted (Number of pages illegally accessed)
      * Location (Location attack came from)
      * WPM_Typing_Speed (Estimated typing speed based on session logs)
      */



   val data= spark.read.option("header", "true").csv("data/hack_data.csv")

    data.printSchema()
    data.show(5)


    data.createOrReplaceTempView("hack_data_table")

    val sqlString="select cast(Session_Connection_Time as Double) ," +
      "cast(Bytes_Transferred  as Double)," +
      "cast(Kali_Trace_Used  as Integer)," +
      "cast(Servers_Corrupted  as Double)," +
      "cast(Pages_Corrupted  as Double)," +
      "cast(Location  as String)," +
      "cast(WPM_Typing_Speed  as Double)" +
      "from hack_data_table";
    println(sqlString)

    val df= spark.sql(sqlString)
    df.show(5)

    val dropLocationDF=df.drop("Location")

    /**
      * Note: The Location column will be useless to consider because
      * the hackers probably used VPNs to hide their real locations during the attacks.
      */

      val cols=Array("Session_Connection_Time", "Bytes Transferred",
        "Kali_Trace_Used",
        "Servers_Corrupted",
        "Pages_Corrupted",
        "WPM_Typing_Speed")
    /**
      * We can assemble our attributes into one column using Spark’s VectorAssembler.
      * When creating a VectorAssembler object, we must specify the input columns and the output column.
      * he input columns are a list of columns that we want to assemble,
      * and the output column is just a name for the column created by the assembler.
      */
   val  assembler = new  VectorAssembler().setInputCols(dropLocationDF.columns).setOutputCol("features")
    /**
      * Now that we’ve created our assembler, we can use it to transform our data. Upon transformation,
      * our data will contain all the original attributes as well as the newly created attribute, called features.
      */

   val  assembled_data = assembler.transform(dropLocationDF)
    println("show assembled_data top 5")

    assembled_data.show(5)

    /**
      * Feature scaling
      */


    //1.Next, we need to standardise our data. To accomplish this, Spark has its own StandardScaler which takes in two arguments
    //  — the name of the input column and the name of the output (scaled) column.

   val  scaler =new   StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")


    /**
      * Let’s fit our scaler to our assembled dataframe and get the final cluster dataset using the .transform() method.
      * After transforming our data, the dataframe will contain the newly created scaledFeatures attribute along with the original attributes.
      */
    val   scaler_model = scaler.fit(assembled_data)
    val   scaled_data = scaler_model.transform(assembled_data)
    println("show scaled_data top 5")
    scaled_data.show(5)
    scaled_data.printSchema()

    /**
      * To tackle the question of whether there were two hackers or three, we can create two k-means models.
      * One model will be initialized with two clusters (k = 2),
      * and the other will be initialized with three clusters (k = 3).
      * We will also specify the column we want to pass in to the model for training.
      */

    val k_means_2 = new KMeans().setFeaturesCol("scaledFeatures").setK(2)
    val k_means_3 = new KMeans().setFeaturesCol("scaledFeatures").setK(3)
    val k_means_5 = new KMeans().setFeaturesCol("scaledFeatures").setK(5)

    /**
      * The idea behind this approach is that based on the number of attacks belonging to each cluster,
      * we can figure out the number of groups involved in the attacks.Let’s fit our models on our scaled_data.
      */
    val  model_k2 = k_means_2.fit(scaled_data)
    val model_k3 = k_means_3.fit(scaled_data)
    val model_k5 = k_means_5.fit(scaled_data)


    /**
      * Was the third hacker involved?
      * Finally, it’s time to find out how many hackers were involved with the attacks.
      * Using .transform() on our clustering models will transform our dataset
      * so that a new attribute called predictions will be created. This new column will contain
      * integers that indicate the cluster to which each attack instance has been classified.
      * Let’s take a look at how many instances are grouped into each cluster in the case of three clusters.
      */
    val model_k3_data = model_k3.transform(scaled_data)
      model_k3_data.groupBy("prediction").count.show


    /**
      * Seems like the number of instances isn’t similar between the three clusters.
      * Since this goes against our background information that the hackers trade off attacks,
      * it seems unlikely that three hackers were involved.Next,
      * let’s take a look at the instance classifications in the case of two clusters.
      */

   val  model_k2_data = model_k2.transform(scaled_data)
     model_k2_data.groupBy("prediction").count.show

    /**
      * Both clusters here have exactly the same number of instances assigned to them,
      * and this perfectly aligns with the idea of hackers trading off attacks.
      * Therefore, it is highly likely that only two hackers were involved with the attacks at RhinoTech.
      */


    // try the k is 5

    val  model_k5_data = model_k5.transform(scaled_data)
    model_k5_data.groupBy("prediction").count.show

  }

}
