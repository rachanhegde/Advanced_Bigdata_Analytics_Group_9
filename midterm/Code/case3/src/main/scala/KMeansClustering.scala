import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object KMeansClustering{

def main(args: Array[String]) {
	// Create new Spark Context
	val conf = new SparkConf().setAppName("KMeansClustering")
	val sc = new SparkContext(conf)

val data1 = MLUtils.loadLibSVMFile(sc, args(0))

val parsedData = data1.map { point =>
(point.features)
}

val numClusters = args(1)
val numIterations = 20
val clusters = KMeans.train(parsedData, numClusters, numIterations)


// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println("Within Set Sum of Squared Errors = " + WSSSE)
	}
}	