STEP 1 : $SPARK_HOME/conf/log4j.properties

		 -- change INFO to ERROR

STEP 2 : Data links for case 1 and 3 are below as the dataset is >100 MB github dosen't allowpushing files larger then 100 MB

		case 1)https://archive.ics.uci.edu/ml/machine-learning-databases/00203/
		case 2)/Case2/data/
		case 3)https://archive.ics.uci.edu/ml/machine-learning-databases/00326/
		 
a) /Case1/classification or regression
	1) sbt package
	2) build.sbt --> change the scala and spark version appropriately (Used SPARK version 1.4.0, Scala Version 2.10.4) 
    3) %SPARK_HOME%/bin/spark-submit --class <Classification-Class-Name> --master local[4] <JAR-File location> <FILE-PATH-LOCATION>
	  
		Classification-Class-Name --  SVMSGDStratifiedSampling, LRStratifiedSampling, LRStratifiedSamplingNFE, LRSGDStratifiedSampling
		
	4)  %SPARK_HOME%/bin/spark-submit --class <Regressionn-Class-Name> --master local[4] <JAR-File location> <FILE-PATH-LOCATION>
		Regressionn-Class-Name -- RegLRSGDStratifiedSampling
		
	5) SPARK - ML -- (Pipeline & cross validation) /Case1/classification or regression/spark-ml ---- classification_ml.scala  to run on spark-shell as sbt package is not compiling for ml

b) /Case2
	1) sbt package
	2) build.sbt --> change the scala and spark version appropriately (Used SPARK version 1.4.0, Scala Version 2.10.4) 
    3) %SPARK_HOME%/bin/spark-submit --class <Classification-Class-Name> --master local[4] <JAR-File location> <FILE-PATH-LOCATION>
	
	Classification-Class-Name -- Case2DecisionTrees
	
	4) SPARK - ML -- (Pipeline & cross validation) /Case1/classification/spark-ml ---- classification_ml.scala  to run on spark-shell as sbt package is not compiling for ml
	
c) 	/Case3
	1) sbt package
	2) build.sbt --> change the scala and spark version appropriately (Used SPARK version 1.4.0, Scala Version 2.10.4) 
    3) %SPARK_HOME%/bin/spark-submit --class <Clustering-Class-Name> --master local[4] <JAR-File location> <FOLDER-LOCATION> <Number-of-Clusters>
	
	Clustering-Class-Name -- KMeansClustering,GMMClustering
    
	FOLDER-LOCATION -- under this keep only libsvm format files
