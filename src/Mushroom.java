import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Mushroom {

    public static void main(String[] args) throws Exception {
        // read datatrain and datatest from file
        Instances datatrain = getDataSet("jamur-train.arff");
        Instances datatest = getDataSet("jamur-test.arff");

//        Classification(datatrain, datatest);
        Clustering(datatrain, datatest);
    }

    private static void Clustering(Instances datatrain, Instances datatest) throws Exception {
        // define clustering algorithm
        SimpleKMeans kmean = new SimpleKMeans();
        kmean.buildClusterer(datatrain);

        // evaluate the clusterer model
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(kmean);
        eval.evaluateClusterer(datatest);

        System.out.println(kmean);
        System.out.println(eval.clusterResultsToString());
    }

    private static void Classification(Instances datatrain, Instances datatest) throws Exception {
        // define naive bayes for classification
        NaiveBayes nb = new NaiveBayes();
        // set the target class for datatrain
        // 0 = classesmushroom {poisonous,edible}
        datatrain.setClassIndex(0);
        nb.buildClassifier(datatrain);

        // define evaluation
        Evaluation eval = new Evaluation(datatrain);
        // set the target class for datatest
        // 0 = classesmushroom {poisonous,edible}
        datatest.setClassIndex(0);
        // evaluate datatrain with data test using target class
        eval.evaluateModel(nb, datatest);

        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
//        System.out.println(eval.toCumulativeMarginDistributionString());
        System.out.println(eval.toMatrixString());
    }

    // get the dataset from datasource
    private static Instances getDataSet(String filepath) throws Exception {
        return new DataSource(filepath).getDataSet();
    }

}
