package hotelbooking;

import java.io.File;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;

public class SimpleStats {

    public static void main(String[] args) throws Exception {
        File dataFile = new File("src/main/resources/data/bd_hotel_bookings.arff");
        if (!dataFile.exists()) {
            System.out.println("Error: bd_hotel_bookings.arff not found. Run MakeArff first!");
            return;
        }

        Instances data = DataSource.read(dataFile.getAbsolutePath());
        int targetIdx = data.attribute("is_canceled") != null ? data.attribute("is_canceled").index() : data.numAttributes() - 1;
        data.setClassIndex(targetIdx);

        System.out.println("==================================================");
        System.out.println("          EXPLORATORY DATA ANALYSIS (EDA)        ");
        System.out.println("==================================================");
        System.out.printf("Total Instances  : %d%n", data.numInstances());
        System.out.printf("Total Attributes : %d%n", data.numAttributes());
        System.out.println("--------------------------------------------------");

        
        System.out.println("\n[1] CLASS BALANCE SUMMARY:");
        AttributeStats classStats = data.attributeStats(data.classIndex());
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            int count = classStats.nominalCounts[i];
            double pct = (count * 100.0) / data.numInstances();
            System.out.printf("  - Class '%s': %d (%.2f%%)%n", data.classAttribute().value(i), count, pct);
        }

        
        System.out.println("\n[2] NUMERIC ATTRIBUTES SUMMARY (Mean, StdDev, Min, Max):");
        System.out.printf("%-30s | %-10s | %-10s | %-10s | %-10s%n", "Attribute", "Mean", "StdDev", "Min", "Max");
        System.out.println("----------------------------------------------------------------------------------");

        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                Stats s = data.attributeStats(i).numericStats;
                if (s != null) {
                    System.out.printf("%-30s | %-10.2f | %-10.2f | %-10.2f | %-10.2f%n",
                            data.attribute(i).name(), s.mean, s.stdDev, s.min, s.max);
                }
            }
        }
        System.out.println("==================================================");
    }
}
