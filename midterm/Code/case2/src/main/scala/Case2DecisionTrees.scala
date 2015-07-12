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
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object Case2DecisionTrees{

def main(args: Array[String]) {
	// Create new Spark Context
	val conf = new SparkConf().setAppName("Case2DecisionTrees")
	val sc = new SparkContext(conf)

//Categorical maps

	
val map1 = Map("Private" -> "0", "Self-emp-not-inc" -> "1", "Self-emp-inc" -> "2", "Federal-gov" -> "3", "Local-gov"-> "4", "State-gov" -> "5","Without-pay" -> "6","Never-worked" -> "7")

val map2 = Map("Bachelors" -> "0", "Some-college" -> "1", "11th" -> "2", "HS-grad" -> "3", "Prof-school"-> "4", "Assoc-acdm" -> "5","Assoc-voc" -> "6", "9th" -> "7", "7th-8th"->"8", "12th"->"9", "Masters" -> "10", "1st-4th"->"11", "10th"->"12", "Doctorate"->"13", "5th-6th"->"14", "Preschool"->"15")

val map3 = Map("Married-civ-spouse" -> "0", "Divorced" -> "1", "Never-married" -> "2", "Separated" -> "3", "Widowed"-> "4", "Married-spouse-absent" -> "5","Married-AF-spouse" -> "6")

val map4 = Map("Tech-support" -> "0", "Craft-repair" -> "1", "Other-service" -> "2", "Sales" -> "3", "Exec-managerial"-> "4", "Prof-specialty" -> "5","Handlers-cleaners" -> "6", "Machine-op-inspct" -> "7", "Adm-clerical" -> "8", "Farming-fishing" -> "9", "Transport-moving" -> "10", "Priv-house-serv" -> "11", "Protective-serv"->"12", "Armed-Forces"->"13")

val map5 = Map("Wife" -> "0", "Own-child" -> "1", "Husband" -> "2", "Not-in-family" -> "3", "Other-relative"-> "4", "Unmarried" -> "5")

val map6 = Map("White" -> "0", "Asian-Pac-Islander" -> "1", "Amer-Indian-Eskimo" -> "2", "Other" -> "3", "Black"-> "4")

val map7 = Map("Female" -> "0", "Male" -> "1")

val map8 = Map("United-States" -> "0", "Cambodia" -> "1", "England" -> "2", "Puerto-Rico" -> "3", "Canada" -> "4", "Germany" -> "5","Outlying-US(Guam-USVI-etc)" -> "6", "India" -> "7", "Japan" -> "8", "Greece" -> "9", "South" -> "10"
, "China" -> "11", "Cuba"->"12", "Iran"->"13", "Honduras"->"14", "Philippines"->"15"
, "Italy" -> "16", "Poland"->"17", "Jamaica"->"18", "Vietnam"->"19", "Mexico"->"20"
, "Portugal" -> "21", "Ireland"->"22", "France"->"23", "Dominican-Republic"->"24", "Laos"->"25"
, "Ecuador" -> "26", "Taiwan"->"27", "Haiti"->"28", "Columbia"->"29", "Hungary"->"30"
, "Guatemala" -> "31", "Nicaragua"->"32", "Scotland"->"33", "Thailand"->"34", "Yugoslavia"->"35"
, "El-Salvador" -> "36", "Trinadad&Tobago"->"37", "Peru"->"38", "Hong"->"39", "Holand-Netherlands"->"40")	
	

val Delimeter = ","
val data = sc.textFile(args(0))
val parsedData = data.map { line =>
val parts = line.split(Delimeter)
val lbl = if (parts(14).contains("<=50K")) 0 else 1
(lbl,parts(0),map1(parts(1).trim()),parts(2),map2(parts(3).trim()),parts(4),map3(parts(5).trim()),map4(parts(6).trim()),map5(parts(7).trim())
,map6(parts(8).trim()),map7(parts(9).trim()),parts(10),parts(11),parts(12),map8(parts(13).trim()))
}

val tmpData = parsedData.map { line =>
LabeledPoint(line._1, Vectors.dense(Array(line._2.toDouble,line._3.toDouble,line._4.toDouble,line._5.toDouble,line._6.toDouble,line._7.toDouble
,line._8.toDouble,line._9.toDouble,line._10.toDouble,line._11.toDouble,line._12.toDouble,line._13.toDouble,line._14.toDouble)))
}

val splits = tmpData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]((1,8),(3,16),(5,7),(6,14),(7,6),(8,5),(9,2))
val impurity = "gini"
val maxDepth = 5
val maxBins = 64

val model = DecisionTree.trainClassifier(training, numClasses, categoricalFeaturesInfo,
  impurity, maxDepth, maxBins)
  
val trainlabelAndPreds = training.map { case LabeledPoint(label, features) =>
	val prediction = model.predict(features)
		(prediction, label)
	}
	
	val trainErr = trainlabelAndPreds.filter(r => r._1 != r._2).count.toDouble / test.count
	val trainMetrics = new MulticlassMetrics(trainlabelAndPreds)
	val btrainMetrics = new BinaryClassificationMetrics(trainlabelAndPreds)
	println("\nTest Error = " + trainErr)
	println("Confusion Matrix: " + trainMetrics.confusionMatrix)
	println("Precision : " + trainMetrics.precision)
	println("Recall : " + trainMetrics.recall)
	println("Precision (1) : " + trainMetrics.precision(1))
	println("Recall (1): " + trainMetrics.recall(1))
	println("Precision (0): " + trainMetrics.precision(0))
	println("Recall (0): " + trainMetrics.recall(0))
	println("Area under ROC: " + btrainMetrics.areaUnderROC)

    val testlabelAndPreds = test.map { case LabeledPoint(label, features) =>
	val prediction = model.predict(features)
		(prediction, label)
	}
	
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

	
	
