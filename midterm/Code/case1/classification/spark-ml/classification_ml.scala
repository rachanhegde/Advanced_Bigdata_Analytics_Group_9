import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.mllib.feature.{StandardScaler,Normalizer,ChiSqSelector}
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import sqlContext.implicits._
import org.apache.spark.rdd.PairRDDFunctions
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.{RegressionMetrics, MulticlassMetrics}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{SQLContext, DataFrame}

object classification_ml{

def main(args: Array[String]) {
	// Create new Spark Context
	val conf = new SparkConf().setAppName("classification_ml")
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

// Training step


// Logistic regression
val lr = new LogisticRegression()

// We may set parameters using setter methods.
lr.setMaxIter(10)
lr.setRegParam(0.01)


// pipeline 

val pipeline = new Pipeline().setStages(Array(lr))

// cross validation

val crossval = new CrossValidator()
crossval.setEstimator(pipeline)
crossval.setEvaluator(new BinaryClassificationEvaluator)

// param grid builder
val paramGrid = new ParamGridBuilder()
paramGrid.addGrid(lr.regParam, Array(0.1, 0.01))
paramGrid.addGrid(lr.maxIter, Array(5, 10, 20))
paramGrid.addGrid(lr.tol, Array(.00001,.001, .0001))
val arrayMap = paramGrid.build()
crossval.setEstimatorParamMaps(arrayMap)
crossval.setNumFolds(4)

val cvModel = crossval.fit(training.toDF)

val predictions = cvModel.transform(training.toDF()).select("prediction", "label")

val predictionsAndLabels = predictions.map {case Row(p: Double, l: Double) => (p, l)}

val trainErr = predictionsAndLabels.filter(r => r._1 != r._2).count.toDouble / training.count


val trainingMetrics = new MulticlassMetrics(predictionsAndLabels)
val bTrainingMetrics = new BinaryClassificationMetrics(predictionsAndLabels)
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



val testPredictions = cvModel.transform(training.toDF()).select("prediction", "label")

val testPredictionsAndLabels = testPredictions.map {case Row(p: Double, l: Double) => (p, l)}

// Get evaluation metrics.
val testErr = testPredictionsAndLabels.filter(r => r._1 != r._2).count.toDouble / test.count
val testMetrics = new MulticlassMetrics(testPredictionsAndLabels)
val bTestMetrics = new BinaryClassificationMetrics(testPredictionsAndLabels)
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