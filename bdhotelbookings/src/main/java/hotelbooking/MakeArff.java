package hotelbooking;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class MakeArff {

    public static void main(String[] args) throws Exception {

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/main/resources/data/bd_hotel_bookings_ml.csv"));

        Instances data = loader.getDataSet();

        StringToNominal stringToNominal = new StringToNominal();
        stringToNominal.setAttributeRange("first-last");
        stringToNominal.setInputFormat(data);
        data = Filter.useFilter(data, stringToNominal);

        int classIndex = data.attribute("is_canceled").index();

        if (data.attribute(classIndex).isNumeric()) {
            NumericToNominal numericToNominal = new NumericToNominal();
            numericToNominal.setAttributeIndices("" + (classIndex + 1));
            numericToNominal.setInputFormat(data);
            data = Filter.useFilter(data, numericToNominal);
        }

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("src/main/resources/data/bd_hotel_bookings.arff"));
        saver.writeBatch();

        System.out.println("ARFF created successfully");
        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());
    }
}