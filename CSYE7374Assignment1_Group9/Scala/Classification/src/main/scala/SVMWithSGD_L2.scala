//SVMWithSGD - L2

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}


object SVMWithSGD_L2{

def main(args: Array[String]) {
	// Create new Spark Context
	val conf = new SparkConf().setAppName("SVMWithSGD_L2")
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

// SVM with defaut L2 regularization
val svmAlg = new SVMWithSGD()
svmAlg.optimizer.
setNumIterations(200)
.setStepSize(.00001)
svmAlg.setIntercept(true)
val model = svmAlg.run(training)


// Training Data Prediction & Score

val labelAndPreds = training.map { case LabeledPoint(label, features) =>
val prediction = model.predict(features)
(prediction, label)
}

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / training.count
val trainingMetrics = new MulticlassMetrics(labelAndPreds)
val bTrainingMetrics = new BinaryClassificationMetrics(labelAndPreds)


// Get evaluation metrics.
println("\nTraining Error = " + trainErr)
println("Confusion Matrix: " + trainingMetrics.confusionMatrix)
println("Precision : " + trainingMetrics.precision)
println("Recall : " + trainingMetrics.recall)
println("Precision (1) : " + trainingMetrics.precision(1))
println("Recall (1): " + trainingMetrics.recall(1))
println("Precision (0): " + trainingMetrics.precision(0))
println("Recall (0): " + trainingMetrics.recall(0))
println("Area under ROC: " + bTrainingMetrics.areaUnderROC)

// Test Data Prediction & Score

val testlabelAndPreds = test.map { case LabeledPoint(label, features) =>
val prediction = model.predict(features)
(prediction, label)
}

// Get evaluation metrics.
val testErr = testlabelAndPreds.filter(r => r._1 != r._2).count.toDouble / test.count
val testMetrics = new MulticlassMetrics(testlabelAndPreds)
val bTestMetrics = new BinaryClassificationMetrics(testlabelAndPreds)
println("\nTest Error = " + testErr)
println("Confusion Matrix: " + testMetrics.confusionMatrix)
println("Precision : " + testMetrics.precision)
println("Recall : " + testMetrics.recall)
println("Precision (1) : " + trainingMetrics.precision(1))
println("Recall (1): " + trainingMetrics.recall(1))
println("Precision (0): " + trainingMetrics.precision(0))
println("Recall (0): " + trainingMetrics.recall(0))
println("Area under ROC: " + bTestMetrics.areaUnderROC)




}
}
