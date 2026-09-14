package hotelbooking;

import java.io.File;
import java.io.FileOutputStream;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToNominal;

public class MakeArff {

    public static void main(String[] args) throws Exception {

        File csvFile = new File("src/main/resources/data/bd_hotel_bookings_ml.csv");
        if (!csvFile.exists()) {
            System.out.println("Error: CSV file not found at " + csvFile.getAbsolutePath());
            return;
        }

        CSVLoader loader = new CSVLoader();
        loader.setSource(csvFile);
        Instances data = loader.getDataSet();

        // Convert String to Nominal
        StringToNominal stringToNominal = new StringToNominal();
        stringToNominal.setAttributeRange("first-last");
        stringToNominal.setInputFormat(data);
        data = Filter.useFilter(data, stringToNominal);

        // Convert target class to Nominal if it is numeric
        int classIndex = data.attribute("is_canceled") != null ? data.attribute("is_canceled").index() : -1;
        if (classIndex != -1 && data.attribute(classIndex).isNumeric()) {
            NumericToNominal numericToNominal = new NumericToNominal();
            numericToNominal.setAttributeIndices("" + (classIndex + 1));
            numericToNominal.setInputFormat(data);
            data = Filter.useFilter(data, numericToNominal);
        }

        
        File targetFile = new File("src/main/resources/data/bd_hotel_bookings.arff");

        
        try (FileOutputStream fos = new FileOutputStream(targetFile)) {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setDestination(fos);
            saver.writeBatch();
        }

        System.out.println("ARFF created successfully");
        System.out.println("Instances  : " + data.numInstances());
        System.out.println("Attributes : " + data.numAttributes());
    }
}