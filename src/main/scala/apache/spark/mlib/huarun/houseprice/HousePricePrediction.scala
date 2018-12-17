package apache.spark.mlib.huarun.houseprice
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{abs, col}

object HousePricePrediction {
  /**
    * https://www.kaggle.com/vikrishnan/boston-house-prices
    * Case class is one of mapping incoming data onto the DataFrame columns
    * https://github.com/jesus-a-martinez-v/boston-housing-spark-mllib/blob/master/src/main/scala/boston_housing_spark_mllib.ipynb
    * http://datasmarts.net/2017/12/17/predicting-boston-housing-prices-in-jupyter-with-spark-mllib/
    *
    * base:https://baijiahao.baidu.com/s?id=1590118860137558338&wfr=spider&for=pc
    */
  case class X(
                id: String ,
                price: Double,
                lotsize: Double,
                bedrooms: Double,
                bathrms: Double,//浴室
                stories: Double,//楼层
                driveway: String,//车道
                recroom: String,//娱乐室
                fullbase: String, //完整度
                gashw: String, //裂痕
                airco: String, //空调
                garagepl: Double, //车库
                prefarea: String)//前置区
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("HousePricePrediction")
      .master("local")
      .getOrCreate()

    spark.sparkContext.setLogLevel("warn")

    import spark.implicits._


    /**
      * Create dataframe using csv method.
      */
    var dataset =  "data/Housing.csv"
    val data = spark.sparkContext.textFile(dataset).
      map(_.split(",")).
      map( x => ( X(
        x(0), x(1).toDouble, x(2).toDouble, x(3).toDouble, x(4).toDouble, x(5).toDouble,
        x(6), x(7), x(8), x(9), x(10), x(11).toDouble, x(12) ))).
      toDF()

    data.show(20)


    /**
      * Define and Identify the Categorical variables
      *
      * To reduce dimensionality, we can separate the numerical and categorical variables
      * and remove the correlated variables. For numerical variables,
      * we’ll use correlation. For categorical variables, we’ll use chi-square test.
      * 统计学概念，定性变量（qualitative variable）又名分类变量 ( categorical variable ):
      * 观测的个体只能归属于几种互不相容类别中的一种时，
      * 一般是用非数字来表达其类别，这样的观测数据称为定性变量。可以理解成可以分类别的变量，如学历、性别、婚否等。
      */
    val categoricalVariables = Array("driveway","recroom", "fullbase", "gashw", "airco", "prefarea")

    val categoricalIndexers: Array[org.apache.spark.ml.PipelineStage] =
      categoricalVariables.map(i => new StringIndexer().setInputCol(i).setOutputCol(i+"Index"))

    /**
      * Initialize the OneHotEncoder as another pipeline stage
      */

    val categoricalEncoders: Array[org.apache.spark.ml.PipelineStage] =
      categoricalVariables.map(e => new OneHotEncoder().
        setInputCol(e + "Index").setOutputCol(e + "Vec"))

    /**
      * Put all the feature columns of the categorical variables together
      */

    val assembler = new VectorAssembler().
      setInputCols( Array(
        "lotsize", "bedrooms", "bathrms", "stories",
        "garagepl","drivewayVec", "recroomVec", "fullbaseVec",
        "gashwVec","aircoVec", "prefareaVec")).
      setOutputCol("features")

    /**
      * Initialize the instance for LinearRegression using your choice of solver and number of iterations
      * Experiment with intercepts and different values of regularization parameter
      */


    val lr = new LinearRegression().
      setLabelCol("price").
      setFeaturesCol("features").
      setRegParam(0.1).
      setMaxIter(100).
      setSolver("l-bfgs")//默认是梯度下降算法

    /**
      * 机器学习中经常利用梯度下降法求最优解问题，通过大量的迭代来得到最优解，
      * 但是对于维度较多的数据，除了占用大量的内存还会很耗时，L-BFGS算法是一种在牛顿法基础上提出的一种求解函数根的算法，下面由简入深尽量用简洁的语言剖析算法的本
      */

    /**
      * Gather the steps and create the pipeline
      */
    val steps = categoricalIndexers ++ categoricalEncoders ++ Array(assembler, lr)

    val pipeline = new Pipeline().setStages(steps)

    /**
      * Split the data into training and test
      */
    val Array(training, test) = data.randomSplit(Array(0.75, 0.25), seed = 12345)

    /**
      * Fit the model and print out the result
      */

    val model = pipeline.fit {
      training
    }

    val holdout = model.transform(test)
    holdout.show(20)

    val prediction = holdout.select("prediction", "price").orderBy(abs(col("prediction")-col("price")))
    prediction.show(20)


    val rm = new RegressionMetrics(prediction.rdd.map{
      x =>  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])
    })
    //https://blog.csdn.net/skullFang/article/details/79107127
   //˙ https://blog.csdn.net/faithmy509/article/details/81217417
    println(s"RMSE = ${rm.rootMeanSquaredError}")//均方根误差
    println(s"R-squared = ${rm.r2}")

// 分清楚RMSE和R-squared



  }

}
