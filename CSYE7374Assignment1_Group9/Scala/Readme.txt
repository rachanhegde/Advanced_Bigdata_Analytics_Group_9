1) Classification mlib

%SPARK_HOME%/bin/spark-submit --class "<CLASS NAME>" --master local[4] <UNZIPPED Assignment1 Folder>Scala\Classification\target\scala-2.11\scala_classifaction_algorithms_2.11-1.0.jar <<winequality-white_SetClassification.csv>>

CLASS NAMES :

a) LogisticRegressionWithLBFGS_L1
b) LogisticRegressionWithLBFGS_L2
c) LogisticRegressionWithSGD_L1
d) LogisticRegressionWithSGD_L2
e) SVMWithSGD_L1
f) SVMWithSGD_L2


2) Regression mlib

%SPARK_HOME%/bin/spark-submit --class "<CLASS NAME>" --master local[4] <UNZIPPED Assignment1 Folder>Scala\Regression\target\scala-2.11\scala_regressions_algorithms_2.11-1.0.jar <<winequality-white_SetRegression.csv>>

CLASS NAMES :

a) LassoWithSGD_L1
b) LinearRegressionWithSGD_L0
c) LinearRegressionWithSGD_L1
d) LinearRegressionWithSGD_L2
e) RidgeRegressionWithSGD_L2





