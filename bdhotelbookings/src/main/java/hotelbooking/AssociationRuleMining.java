package hotelbooking;

import java.io.File;
import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class AssociationRuleMining {

    
    public static String formatIntervals(String text) {
        if (text == null) return "";
        // '(-inf-X]' -> <= X
        String cleaned = text.replaceAll("'\\(-inf-([0-9.]+)\\]'", "<= $1");
        // '(X-inf)' or '(X-inf)' -> > X
        cleaned = cleaned.replaceAll("'\\(([0-9.]+)-inf\\)'", "> $1");
        cleaned = cleaned.replaceAll("'\\(([0-9.]+)-inf\\]'", "> $1");
        // '(X-Y]' -> (X to Y)
        cleaned = cleaned.replaceAll("'\\(([0-9.]+)-([0-9.]+)\\]'", "($1 to $2)");
        return cleaned;
    }

    public static void main(String[] args) throws Exception {
        File dataFile = new File("src/main/resources/data/bd_hotel_bookings.arff");
        if (!dataFile.exists()) {
            System.out.println("Error: bd_hotel_bookings.arff not found. Run MakeArff first!");
            return;
        }

        Instances data = DataSource.read(dataFile.getAbsolutePath());

        
        String[] keepAttrs = {
            "hotel", "is_canceled", "lead_time", "deposit_type", 
            "customer_type", "market_segment", "previous_cancellations", "total_of_special_requests"
        };
        
        StringBuilder keepIndices = new StringBuilder();
        for (String name : keepAttrs) {
            if (data.attribute(name) != null) {
                keepIndices.append(data.attribute(name).index() + 1).append(",");
            }
        }
        
        Remove removeNoise = new Remove();
        removeNoise.setAttributeIndices(keepIndices.toString());
        removeNoise.setInvertSelection(true);
        removeNoise.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, removeNoise);

        
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(filteredData);
        filteredData = Filter.useFilter(filteredData, replaceMissing);

        
        Discretize discretize = new Discretize();
        discretize.setBins(3);
        discretize.setUseEqualFrequency(true);
        discretize.setInputFormat(filteredData);
        Instances nominalData = Filter.useFilter(filteredData, discretize);

        
        int classIdx = nominalData.attribute("is_canceled") != null ? 
                       nominalData.attribute("is_canceled").index() : nominalData.numAttributes() - 1;
        nominalData.setClassIndex(classIdx);

        
        System.out.println("==================================================");
        System.out.println("  PART A: GENERAL TELEMETRY / FEATURE RULES       ");
        System.out.println("==================================================");

        Apriori generalApriori = new Apriori();
        generalApriori.setNumRules(10);
        generalApriori.setLowerBoundMinSupport(0.10);
        generalApriori.setMinMetric(1.2);
        generalApriori.setMetricType(new SelectedTag(1, Apriori.TAGS_SELECTION)); // 1 = Lift
        generalApriori.buildAssociations(nominalData);

        
        System.out.println(formatIntervals(generalApriori.toString()));

        
        System.out.println("==================================================");
        System.out.println("  PART B: CLASS ASSOCIATION RULES (PREDICTING CANCELLATION)");
        System.out.println("==================================================");

        Apriori classApriori = new Apriori();
        classApriori.setNumRules(15);
        classApriori.setClassIndex(classIdx + 1); 
        classApriori.setCar(true);
        classApriori.setLowerBoundMinSupport(0.02);
        classApriori.setUpperBoundMinSupport(0.50);
        classApriori.setMetricType(new SelectedTag(0, Apriori.TAGS_SELECTION)); 
        classApriori.setMinMetric(0.60);
        classApriori.buildAssociations(nominalData);

        
        System.out.println(formatIntervals(classApriori.toString()));

        System.out.println("==================================================");
    }
}