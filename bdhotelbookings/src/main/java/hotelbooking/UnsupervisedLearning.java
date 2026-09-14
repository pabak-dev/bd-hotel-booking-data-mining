package hotelbooking;

import java.io.File;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class UnsupervisedLearning {

    public static void main(String[] args) throws Exception {
        File dataFile = new File("src/main/resources/data/bd_hotel_bookings.arff");
        if (!dataFile.exists()) {
            System.out.println("Error: bd_hotel_bookings.arff not found. Run MakeArff first!");
            return;
        }

        Instances data = DataSource.read(dataFile.getAbsolutePath());
        int targetIdx = data.attribute("is_canceled") != null ? data.attribute("is_canceled").index() : data.numAttributes() - 1;
        data.setClassIndex(targetIdx);

        
        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        replaceMissing.setInputFormat(data);
        Instances cleanData = Filter.useFilter(data, replaceMissing);

        Normalize normalize = new Normalize();
        normalize.setInputFormat(cleanData);
        cleanData = Filter.useFilter(cleanData, normalize);

        
        Remove removeClass = new Remove();
        removeClass.setAttributeIndices("" + (cleanData.classIndex() + 1));
        removeClass.setInputFormat(cleanData);
        Instances clusterInput = Filter.useFilter(cleanData, removeClass);

        
        int numClusters = 3;
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClusters);
        kMeans.setSeed(42);
        kMeans.setPreserveInstancesOrder(true);
        kMeans.setDistanceFunction(new EuclideanDistance());
        kMeans.buildClusterer(clusterInput);

        System.out.println("==================================================");
        System.out.println("          K-MEANS CLUSTERING RESULTS              ");
        System.out.println("==================================================");
        System.out.println("Number of Clusters: " + numClusters);
        System.out.println("Squared Error     : " + kMeans.getSquaredError());

        
        int[][] clusterToClass = new int[numClusters][data.classAttribute().numValues()];
        double[] clusterSizes = kMeans.getClusterSizes();

        for (int i = 0; i < data.numInstances(); i++) {
            int clusterNum = kMeans.clusterInstance(clusterInput.instance(i));
            int actualClass = (int) data.instance(i).classValue();
            clusterToClass[clusterNum][actualClass]++;
        }

        System.out.println("\nCluster vs Target Class Breakdown:");
        System.out.printf("%-12s | %-10s |", "Cluster #", "Size");
        for (int c = 0; c < data.classAttribute().numValues(); c++) {
            System.out.printf(" %-15s |", data.classAttribute().value(c));
        }
        System.out.println();
        System.out.println("-------------------------------------------------------------------");

        for (int k = 0; k < numClusters; k++) {
            int size = (int) clusterSizes[k];
            System.out.printf("Cluster %-4d | %-10d |", k, size);
            for (int c = 0; c < data.classAttribute().numValues(); c++) {
                double pct = size > 0 ? (clusterToClass[k][c] * 100.0) / size : 0.0;
                System.out.printf(" %-5d (%5.1f%%)   |", clusterToClass[k][c], pct);
            }
            System.out.println();
        }

        
        System.out.println("\n==================================================");
        System.out.println("         CLUSTER CENTROIDS (CHARACTERISTICS)       ");
        System.out.println("==================================================");

        Instances centroids = kMeans.getClusterCentroids();
        System.out.printf("%-32s |", "Attribute");
        for (int k = 0; k < numClusters; k++) {
            System.out.printf(" Cluster %-10d |", k);
        }
        System.out.println();
        System.out.println("----------------------------------------------------------------------------------");

        for (int j = 0; j < centroids.numAttributes(); j++) {
            Attribute attr = centroids.attribute(j);
            System.out.printf("%-32s |", attr.name());

            for (int k = 0; k < numClusters; k++) {
                if (attr.isNominal()) {
                    System.out.printf(" %-18s |", centroids.instance(k).stringValue(j));
                } else {
                    System.out.printf(" %-18.4f |", centroids.instance(k).value(j));
                }
            }
            System.out.println();
        }
        System.out.println("==================================================");
    }
}