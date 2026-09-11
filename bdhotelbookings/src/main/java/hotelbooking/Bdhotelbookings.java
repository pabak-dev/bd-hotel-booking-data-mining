package hotelbooking;

import java.util.Scanner;

public class Bdhotelbookings {

    public static void main(String[] args) {

        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.println();
            System.out.println("========================================");
            System.out.println(" BANGLADESH HOTEL BOOKING DATA MINING");
            System.out.println("========================================");
            System.out.println("1. Convert CSV to ARFF");
            System.out.println("2. Run Supervised Learning");
            System.out.println("3. Run Full Pipeline");
            System.out.println("0. Exit");
            System.out.println("========================================");
            System.out.print("Select option: ");

            String choice = scanner.nextLine();

            try {

                switch (choice) {

                    case "1":
                        System.out.println();
                        System.out.println("Creating ARFF...");
                        MakeArff.main(new String[0]);
                        break;

                    case "2":
                        System.out.println();
                        System.out.println("Running Supervised Learning...");
                        SupervisedLearning.main(new String[0]);
                        break;

                    case "3":
                        System.out.println();
                        System.out.println("Creating ARFF...");
                        MakeArff.main(new String[0]);

                        System.out.println();
                        System.out.println("Running Supervised Learning...");
                        SupervisedLearning.main(new String[0]);

                        System.out.println();
                        System.out.println("Full pipeline completed.");
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
            }
        }
    }
}