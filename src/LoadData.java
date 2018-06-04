import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadData {

    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data j2ajmur33333.csv");
        Instances data = source.getDataSet();
//        System.out.println(data.numAttributes());
//        System.out.println(data.toString());


    }

}
