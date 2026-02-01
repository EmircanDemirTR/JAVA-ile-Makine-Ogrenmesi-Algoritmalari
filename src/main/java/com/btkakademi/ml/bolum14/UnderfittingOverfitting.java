package com.btkakademi.ml.bolum14;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;
import smile.regression.RegressionTree;
import smile.util.Index;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

public class UnderfittingOverfitting {
    static void main() {
        try {
            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader().setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true).setTrim(true)
                    .build();

            var url = UnderfittingOverfitting.class.getClassLoader().getResource("datasets/auto-mpg.csv");
            DataFrame veri = Read.csv(Path.of(url.toURI()), format);

            // Train Test Split
            int n = veri.nrow();
            int egitimBoyut = (int) (n * 0.7);
            int testBoyut = n - egitimBoyut;

            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) {
                idx[i] = i;
            }
            Collections.shuffle(Arrays.asList(idx), new Random(42));

            int[] trainIdx = new int[egitimBoyut];
            int[] testIdx = new int[testBoyut];
            for (int i = 0; i < egitimBoyut; i++) {
                trainIdx[i] = idx[i];
            }
            for (int i = 0; i < testBoyut; i++) {
                testIdx[i] = idx[i + egitimBoyut];
            }
            DataFrame trainVeri = veri.get(Index.of(trainIdx));
            DataFrame testVeri = veri.get(Index.of(testIdx));

            System.out.println("Egitim seti: " + trainVeri.nrow());
            System.out.println("Test seti: " + testVeri.nrow());


            // Underfitting Demo
            // Sadece weight ve mpg sutunlarından yeni DataFrame oluştur
            double[][] tekOzellikData = new double[n][2];
            for (int i = 0; i < n; i++) {
                tekOzellikData[i][0] = veri.getDouble(i, 3); // weight
                tekOzellikData[i][1] = veri.getDouble(i, 5); // mpg
            }
            DataFrame tekOzellikVeri = DataFrame.of(tekOzellikData, "weight", "mpg");

            DataFrame tekTrain = tekOzellikVeri.get(Index.of(trainIdx));
            DataFrame tekTest = tekOzellikVeri.get(Index.of(testIdx));

            // OLS ile Basit Doğrusal Regresyon Modeli
            Formula formulaTek = Formula.lhs("mpg");
            LinearModel basitModel = OLS.fit(formulaTek, tekTrain);

            double[] basitTrainMetrik = hesaplaMetrikler(basitModel, tekTrain, "mpg");
            double[] basitTestMetrik = hesaplaMetrikler(basitModel, tekTest, "mpg");
            System.out.println("\n--- Basit Doğrusal Regresyon Modeli (Underfitting) ---");
            System.out.printf(" Train RMSE: %.4f  |  Test RMSE: %.4f%n", basitTrainMetrik[0], basitTestMetrik[0]);
            System.out.printf(" Train R2:   %.4f  |  Test R2:   %.4f%n", basitTrainMetrik[1], basitTestMetrik[1]);
            System.out.printf(" Train MAE:  %.4f  |  Test MAE:  %.4f%n", basitTrainMetrik[2], basitTestMetrik[2]);

            // Normal Model
            Formula formulaTum = Formula.lhs("mpg");
            LinearModel uygunModel = OLS.fit(formulaTum, trainVeri);

            double[] uygunTrainMetrik = hesaplaMetrikler(uygunModel, trainVeri, "mpg");
            double[] uygunTestMetrik = hesaplaMetrikler(uygunModel, testVeri, "mpg");
            System.out.println("\n--- Uygun Model ---");
            System.out.printf(" Train RMSE: %.4f  |  Test RMSE: %.4f%n", uygunTrainMetrik[0], uygunTestMetrik[0]);
            System.out.printf(" Train R2:   %.4f  |  Test R2:   %.4f%n", uygunTrainMetrik[1], uygunTestMetrik[1]);
            System.out.printf(" Train MAE:  %.4f  |  Test MAE:  %.4f%n", uygunTrainMetrik[2], uygunTestMetrik[2]);

            // Model Katsayıları
            System.out.println("\nModel Katsayıları:");
            String[] ozellikler = {"cylinders", "displacement", "horsepower", "weight", "acceleration"};
            var katsayilar = uygunModel.coefficients();
            for (int i = 0; i < ozellikler.length; i++) {
                System.out.printf("  %s: %.6f%n", ozellikler[i], katsayilar.get(i));
            }
            System.out.printf("  Sabit Terim: %.6f%n", uygunModel.intercept());


            // Overfitting Demo
            System.out.println("\n--- Overfitting Demo ---");
            var overFitOptions = new RegressionTree.Options(100, 0, 1);
            RegressionTree overFitModel = RegressionTree.fit(formulaTum, trainVeri, overFitOptions);

            double[] overFitTrainMetrik = hesaplaAgacMetrikler(overFitModel, trainVeri, "mpg");
            double[] overFitTestMetrik = hesaplaAgacMetrikler(overFitModel, testVeri, "mpg");
            System.out.printf(" Train RMSE: %.4f  |  Test RMSE: %.4f%n", overFitTrainMetrik[0], overFitTestMetrik[0]);
            System.out.printf(" Train R2:   %.4f  |  Test R2:   %.4f%n", overFitTrainMetrik[1], overFitTestMetrik[1]);
            System.out.printf(" Train MAE:  %.4f  |  Test MAE:  %.4f%n", overFitTrainMetrik[2], overFitTestMetrik[2]);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[] hesaplaMetrikler(LinearModel model, DataFrame veri, String hedef) {
        int n = veri.nrow();
        int hedefIdx = veri.schema().indexOf(hedef);
        double toplamKareHata = 0, toplamKareSapma = 0, toplamMutlakHata = 0, toplam = 0;

        for (int i = 0; i < n; i++) toplam += veri.getDouble(i, hedefIdx);
        double ortalama = toplam / n;

        for (int i = 0; i < n; i++) {
            double gercek = veri.getDouble(i, hedefIdx);
            double tahmin = model.predict(veri.get(i));
            double hata = gercek - tahmin;

            toplamKareHata += hata * hata;
            toplamMutlakHata += Math.abs(hata);
            double sapma = gercek - ortalama;
            toplamKareSapma += sapma * sapma;
        }

        double rmse = Math.sqrt(toplamKareHata / n);
        double r2 = 1 - (toplamKareHata / toplamKareSapma);
        double mae = toplamMutlakHata / n;
        return new double[]{rmse, r2, mae};
    }

    private static double[] hesaplaAgacMetrikler(RegressionTree model, DataFrame veri, String hedef) {
        int n = veri.nrow();
        int hedefIdx = veri.schema().indexOf(hedef);
        double toplamKareHata = 0, toplamKareSapma = 0, toplamMutlakHata = 0, toplam = 0;

        for (int i = 0; i < n; i++) toplam += veri.getDouble(i, hedefIdx);
        double ortalama = toplam / n;

        for (int i = 0; i < n; i++) {
            double gercek = veri.getDouble(i, hedefIdx);
            double tahmin = model.predict(veri.get(i));
            double hata = gercek - tahmin;

            toplamKareHata += hata * hata;
            toplamMutlakHata += Math.abs(hata);
            double sapma = gercek - ortalama;
            toplamKareSapma += sapma * sapma;
        }

        double rmse = Math.sqrt(toplamKareHata / n);
        double r2 = 1 - (toplamKareHata / toplamKareSapma);
        double mae = toplamMutlakHata / n;
        return new double[]{rmse, r2, mae};
    }
}
