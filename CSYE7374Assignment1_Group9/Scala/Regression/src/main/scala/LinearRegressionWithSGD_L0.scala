import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.RegressionMetrics

object LinearRegressionWithSGD_L0{

def main(args: Array[String]) {
// Create new Spark Context
	val conf = new SparkConf().setAppName("LinearRegressionWithSGD_L0")
	val sc = new SparkContext(conf)
	
	// Load and parse the data file
	val Delimeter = ","
	val data = sc.textFile(args(0))
	val parsedData = data.map { line =>
	val parts = line.split(Delimeter)
	LabeledPoint(parts(0).toDouble, Vectors.dense(parts.slice(1,11).map(x => x.toDouble).toArray))
	}
	val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)




// LR default L2
val lrAlg = new LinearRegressionWithSGD()
lrAlg.optimizer
.setNumIterations(100)
.setStepSize(0.001)	
lrAlg.setIntercept(true)
val model = lrAlg.run(training)



// Training Data Prediction & Score

val labelAndPreds = training.map { point =>
val prediction = model.predict(point.features)
(point.label, prediction)
}


val trainingMetrics = new RegressionMetrics(labelAndPreds)
// Get evaluation metrics.
println("\nRoot Mean Squared Error = " + trainingMetrics.rootMeanSquaredError)


// Test Data Prediction & Score

val testlabelAndPreds = test.map { case LabeledPoint(label, features) =>
val prediction = model.predict(features)
(prediction, label)
}

// Get evaluation metrics.
val testMetrics = new RegressionMetrics(testlabelAndPreds)
println("Root Mean Squared Error : " + testMetrics.rootMeanSquaredError)

}
}


