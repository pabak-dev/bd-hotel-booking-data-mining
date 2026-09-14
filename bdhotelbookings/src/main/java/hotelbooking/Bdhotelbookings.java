package hotelbooking;

import java.util.Scanner;

public class Bdhotelbookings {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.println();
            System.out.println("========================================");
            System.out.println("   DATA MINING & PREDICTIVE PIPELINE    ");
            System.out.println("========================================");
            System.out.println("1. Convert CSV to ARFF (Data Prep)");
            System.out.println("2. Exploratory Data Analysis (Simple Stats)");
            System.out.println("3. Run Supervised Learning");
            System.out.println("4. Run Unsupervised Learning (K-Means)");
            System.out.println("5. Mine Association Rules (Apriori)");
            System.out.println("6. Run Full Pipeline");
            System.out.println("0. Exit");
            System.out.println("========================================");
            System.out.print("Select option: ");

            String choice = scanner.nextLine();

            try {
                switch (choice) {
                    case "1":
                        System.out.println("\nCreating ARFF...");
                        MakeArff.main(new String[0]);
                        break;

                    case "2":
                        SimpleStats.main(new String[0]);
                        break;

                    case "3":
                        System.out.println("\nRunning Supervised Learning...");
                        SupervisedLearning.main(new String[0]);
                        break;

                    case "4":
                        System.out.println("\nRunning Unsupervised Learning...");
                        UnsupervisedLearning.main(new String[0]);
                        break;

                    case "5":
                        System.out.println("\nRunning Association Rule Mining...");
                        AssociationRuleMining.main(new String[0]);
                        break;

                    case "6":
                        System.out.println("\n--- Step 1: Data Preparation ---");
                        MakeArff.main(new String[0]);

                        System.out.println("\n--- Step 2: Simple Stats ---");
                        SimpleStats.main(new String[0]);

                        System.out.println("\n--- Step 3: Supervised Learning ---");
                        SupervisedLearning.main(new String[0]);

                        System.out.println("\n--- Step 4: Unsupervised Learning ---");
                        UnsupervisedLearning.main(new String[0]);

                        System.out.println("\n--- Step 5: Association Rule Mining ---");
                        AssociationRuleMining.main(new String[0]);

                        System.out.println("\nFull pipeline completed successfully.");
                        break;

                    case "0":
                        System.out.println("Program closed.");
                        scanner.close();
                        return;

                    default:
                        System.out.println("Invalid option.");
                }
            } catch (Exception e) {
                System.out.println("Error: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
}