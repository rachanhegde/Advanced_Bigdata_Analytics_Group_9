import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.GaussianMixture
import org.apache.spark.mllib.clustering.GaussianMixtureModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

object GMMClustering{

def main(args: Array[String]) {
	// Create new Spark Context
	val conf = new SparkConf().setAppName("GMMClustering")
	val sc = new SparkContext(conf)

// Load and parse the data
val data1 = MLUtils.loadLibSVMFile(sc, args(0))

val parsedData = data1.map { point =>
(point.features)
}

// Cluster the data into two classes using GaussianMixture
val gmm = new GaussianMixture().setK(args(1)).run(parsedData)

// Save and load model
gmm.save(sc, "myGMMModel")
val sameModel = GaussianMixtureModel.load(sc, "myGMMModel")

// output parameters of max-likelihood model
for (i <- 0 until gmm.k) {
  println("weight=%f\nmu=%s\nsigma=\n%s\n" format
    (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
}
	}
}	