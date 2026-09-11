package hotelbooking;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class SupervisedLearning {

    public static FilteredClassifier createModel(Classifier classifier) throws Exception {

        ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
        Normalize normalize = new Normalize();

        MultiFilter filters = new MultiFilter();
        filters.setFilters(new weka.filters.Filter[]{
            replaceMissing,
            normalize
        });

        FilteredClassifier model = new FilteredClassifier();
        model.setFilter(filters);
        model.setClassifier(classifier);

        return model;
    }

    public static void main(String[] args) throws Exception {

        Instances data = DataSource.read(
                "src/main/resources/data/bd_hotel_bookings.arff"
        );

        data.setClassIndex(data.attribute("is_canceled").index());

        int canceledIndex = data.classAttribute().indexOfValue("1");

        System.out.println("Instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());
        System.out.println("Class: " + data.classAttribute().name());

        int canceled = 0;
        int notCanceled = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            if ((int) data.instance(i).classValue() == canceledIndex) {
                canceled++;
            } else {
                notCanceled++;
            }
        }

        System.out.println("Canceled: " + canceled);
        System.out.println("Not Canceled: " + notCanceled);

        Classifier[] classifiers = {
            new RandomForest(),
            new J48(),
            new NaiveBayes()
        };

        String[] names = {
            "Random Forest",
            "J48 Decision Tree",
            "Naive Bayes"
        };

        double bestAUC = -1;
        String bestName = "";
        Classifier bestClassifier = null;

        for (int i = 0; i < classifiers.length; i++) {

            FilteredClassifier model = createModel(classifiers[i]);

            Evaluation evaluation = new Evaluation(data);

            evaluation.crossValidateModel(
                    model,
                    data,
                    10,
                    new Random(42)
            );

            double accuracy = evaluation.pctCorrect();
            double precision = evaluation.precision(canceledIndex);
            double recall = evaluation.recall(canceledIndex);
            double f1 = evaluation.fMeasure(canceledIndex);
            double auc = evaluation.areaUnderROC(canceledIndex);

            System.out.println();
            System.out.println("===== " + names[i] + " =====");
            System.out.printf("Accuracy: %.2f%%%n", accuracy);
            System.out.printf("Precision: %.4f%n", precision);
            System.out.printf("Recall: %.4f%n", recall);
            System.out.printf("F1 Score: %.4f%n", f1);
            System.out.printf("ROC AUC: %.4f%n", auc);

            System.out.println(
                    evaluation.toMatrixString("Confusion Matrix")
            );

            if (auc > bestAUC) {
                bestAUC = auc;
                bestName = names[i];
                bestClassifier = classifiers[i];
            }
        }

        FilteredClassifier finalModel = createModel(bestClassifier);
        finalModel.buildClassifier(data);

        SerializationHelper.write(
                "best_hotel_cancellation_model.model",
                finalModel
        );

        Classifier loadedModel =
                (Classifier) SerializationHelper.read(
                        "best_hotel_cancellation_model.model"
                );

        System.out.println();
        System.out.println("===== FINAL RESULT =====");
        System.out.println("Best Model: " + bestName);
        System.out.printf("Best ROC AUC: %.4f%n", bestAUC);
        System.out.println("Model saved successfully");
        System.out.println(
                "Loaded Model: " + loadedModel.getClass().getSimpleName()
        );
    }
}