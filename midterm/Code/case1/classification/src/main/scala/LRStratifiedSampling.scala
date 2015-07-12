import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.feature.{StandardScaler,Normalizer,ChiSqSelector}
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.rdd.PairRDDFunctions
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition


object LRStratifiedSampling{

def main(args: Array[String]) {
	// Create new Spark Context
	val conf = new SparkConf().setAppName("LRStratifiedSampling")
	val sc = new SparkContext(conf)

val Delimeter = ","
val data = sc.textFile(args(0))

// Stratified Sampling
val data1 = data.map(l => l.split(",").map(_.toDouble)).map(l => (l.head, Vectors.dense(l.takeRight(88))))
val data2 = data.map{ line =>
val parts = line.split(Delimeter)
val lbl = if (parts(0).toDouble >= 1965) 1 else 0
(lbl, Vectors.dense(parts.slice(1,89).map(x => x.toDouble)))
}
val fractions = Map(0 -> 8.0, 1->1.0)
val stratifiedData = data2.sampleByKey(true, fractions)

val parsedData = stratifiedData.map { line =>
LabeledPoint(line._1, line._2)
}

// Feature Scaling 
// Standard Scalar normalizer
val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedData.map(x => x.features))
val parsedDataNormalized = parsedData.map(x => LabeledPoint(x.label, scaler.transform(x.features)))

//PCA
val pca = new PCA(10).fit(parsedDataNormalized.map(_.features))
val projected = parsedDataNormalized.map(p => p.copy(features = pca.transform(p.features)))

val splits = projected.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

val lrAlg = new LogisticRegressionWithLBFGS()
		lrAlg.optimizer
		.setUpdater(new L1Updater)
		.setConvergenceTol(.00001)
		lrAlg.setNumClasses(2)
		lrAlg.setIntercept(true)
val model = lrAlg.run(training)

val trainlabelAndPreds = training.map { case LabeledPoint(label, features) =>
	val prediction = model.predict(features)
		(prediction, label)
	}
	
	val trainErr = trainlabelAndPreds.filter(r => r._1 != r._2).count.toDouble / training.count
	val trainingMetrics = new MulticlassMetrics(trainlabelAndPreds)
	val bTrainingMetrics = new BinaryClassificationMetrics(trainlabelAndPreds)
	// Get evaluation metrics.	
	println("\nTrain Error = " + trainErr)
	println("Confusion Matrix: " + trainingMetrics.confusionMatrix)
	println("Precision : " + trainingMetrics.precision)
	println("Recall : " + trainingMetrics.recall)
	println("Precision (1) : " + trainingMetrics.precision(1))
	println("Recall (1): " + trainingMetrics.recall(1))
	println("Precision (0): " + trainingMetrics.precision(0))
	println("Recall (0): " + trainingMetrics.recall(0))
	println("Area under ROC: " + bTrainingMetrics.areaUnderROC)

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
	println("Precision (1) : " + testMetrics.precision(1))
	println("Recall (1): " + testMetrics.recall(1))
	println("Precision (0): " + testMetrics.precision(0))
	println("Recall (0): " + testMetrics.recall(0))
	println("Area under ROC: " + bTestMetrics.areaUnderROC)
	}
}	